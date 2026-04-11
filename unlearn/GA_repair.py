import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

import utils
from LS import LabelSmoothingCrossEntropy
from .impl import iterative_unlearn


def _make_anchor(model):
    """Snapshot current (pre-unlearning) parameters for L2 anchoring."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _l2_anchor_loss(model, anchor_state):
    denom = 0.0
    loss = 0.0
    for (n, p) in model.named_parameters():
        if not p.requires_grad:
            continue
        a = anchor_state.get(n, None)
        if a is None:
            continue
        loss = loss + (p - a).pow(2).sum()
        denom += p.numel()
    if denom == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return loss / denom


def _l1_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.abs().sum()
    denom = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / max(1, denom)


def _entropy_mean(logits):
    prob = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    ent = -(prob * logp).sum(dim=1).mean()
    return ent


@iterative_unlearn
def GA_repair(data_loaders, model, criterion, optimizer, epoch, args):
    """
    Alternating unlearning (GA) + repair (retain-side SAM) with parameter anchoring,
    label smoothing, optional Mixup, and entropy maximization on forget batches.
    This improves TA/RA while preserving/boosting MIA forget efficacy.
    """
    device = next(model.parameters()).device

    # ensure train-time augmentation during the unlearning phase
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    utils.dataset_convert_to_train(retain_loader.dataset)
    utils.dataset_convert_to_train(forget_loader.dataset)

    # Initialize (once) state used across epochs
    if not hasattr(model, "_anchor_state"):
        model._anchor_state = _make_anchor(model)

    # --- 1) GA step on a small slice of forget set ---
    ga_opt = getattr(args, "_ga_opt", None)
    if ga_opt is None:
        ga_opt = torch.optim.SGD(
            model.parameters(),
            lr=args.unlearn_lr,
            momentum=getattr(args, "momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 5e-4),
        )
        args._ga_opt = ga_opt

    model.train()
    ga_steps = max(1, int(getattr(args, "forget_steps", 1)))
    start = time.time()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    for bi, (images, targets) in enumerate(forget_loader):
        if bi >= ga_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        ce = F.cross_entropy(logits, targets)

        l1 = _l1_norm(model)
        ent = _entropy_mean(logits)
        anchor_loss = _l2_anchor_loss(model, model._anchor_state)

        loss = -ce + getattr(args, "alpha", 0.2) * l1 + getattr(args, "entropy_beta", 0.5) * ent \
               + getattr(args, "anchor_lambda", 1e-4) * anchor_loss

        ga_opt.zero_grad(set_to_none=True)
        loss.backward()
        ga_opt.step()

        with torch.no_grad():
            prec1 = utils.accuracy(logits.float().data, targets)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    if ga_steps > 0:
        print(f"[GA] epoch {epoch} steps {ga_steps}  loss {losses.avg:.4f}  acc {top1.avg:.2f}  time {time.time()-start:.1f}s")

    # --- 2) Retain-side SAM repair ---
    from SAM import SAM
    base_optimizer = torch.optim.SGD
    repair_opt = getattr(args, "_repair_opt", None)
    if repair_opt is None:
        repair_opt = SAM(
            model.parameters(),
            base_optimizer,
            rho=getattr(args, "sam_rho", 2.0),
            adaptive=True,
            lr=getattr(args, "repair_lr", 0.01),
            momentum=getattr(args, "momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 5e-4),
        )
        args._repair_opt = repair_opt

    ls_eps = getattr(args, "label_smoothing_eps", 0.05)
    ce_ls = LabelSmoothingCrossEntropy(eps=ls_eps) if ls_eps and ls_eps > 0 else criterion

    retain_steps = max(1, int(getattr(args, "retain_steps", 2)))
    start = time.time()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    for bi, (images, targets) in enumerate(retain_loader):
        if bi >= retain_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # optional mixup
        mix_alpha = getattr(args, "mixup_alpha", 0.2)
        if mix_alpha and mix_alpha > 0:
            lam = np.random.beta(mix_alpha, mix_alpha)
            index = torch.randperm(images.size(0), device=device)
            mixed_x = lam * images + (1 - lam) * images[index]
            y_a, y_b = targets, targets[index]

            # first SAM step
            logits = model(mixed_x)
            loss = lam * ce_ls(logits, y_a) + (1 - lam) * ce_ls(logits, y_b) + \
                   getattr(args, "anchor_lambda", 1e-4) * _l2_anchor_loss(model, model._anchor_state)
            repair_opt.zero_grad(set_to_none=True)
            loss.backward()
            repair_opt.first_step(zero_grad=True)

            # second SAM step
            logits2 = model(mixed_x)
            loss2 = lam * ce_ls(logits2, y_a) + (1 - lam) * ce_ls(logits2, y_b) + \
                    getattr(args, "anchor_lambda", 1e-4) * _l2_anchor_loss(model, model._anchor_state)
            loss2.backward()
            repair_opt.second_step(zero_grad=True)

            with torch.no_grad():
                prec1 = utils.accuracy(logits2.float().data, y_a)[0]  # approximate
                losses.update(loss2.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))
        else:
            # No mixup
            logits = model(images)
            loss = ce_ls(logits, targets) + getattr(args, "anchor_lambda", 1e-4) * _l2_anchor_loss(model, model._anchor_state)
            repair_opt.zero_grad(set_to_none=True)
            loss.backward()
            repair_opt.first_step(zero_grad=True)

            logits2 = model(images)
            loss2 = ce_ls(logits2, targets) + getattr(args, "anchor_lambda", 1e-4) * _l2_anchor_loss(model, model._anchor_state)
            loss2.backward()
            repair_opt.second_step(zero_grad=True)

            with torch.no_grad():
                prec1 = utils.accuracy(logits2.float().data, targets)[0]
                losses.update(loss2.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))

    print(f"[Repair] epoch {epoch} steps {retain_steps}  loss {losses.avg:.4f}  acc {top1.avg:.2f}  time {time.time()-start:.1f}s")
    return top1.avg