import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pruner
import utils
from LS import LabelSmoothingCrossEntropy
from SAM import SAM
from .GA_repair import _entropy_mean, _l2_anchor_loss, _make_anchor
from .impl import iterative_unlearn


def _forget_margin_loss(logits, targets, margin):
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    return F.relu(target_logits - other_logits + margin).mean()


def _distill_kl(student_logits, teacher_logits, temperature):
    temp = max(float(temperature), 1e-6)
    student_logp = F.log_softmax(student_logits / temp, dim=1)
    teacher_prob = F.softmax(teacher_logits / temp, dim=1)
    return F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (temp * temp)


def _tensor_is_finite(tensor):
    return torch.isfinite(tensor).all().item()


def _gradients_are_finite(model):
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def _clip_trainable_grads(model, max_norm=5.0):
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def _iter_bn_modules(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module in model.modules():
        if isinstance(module, bn_types):
            yield module


def _freeze_bn_for_stage(model):
    bn_state = []
    for module in _iter_bn_modules(model):
        state = {"module": module, "training": module.training}
        module.eval()
        if getattr(module, "weight", None) is not None:
            state["weight_grad"] = module.weight.requires_grad
            module.weight.requires_grad_(False)
        if getattr(module, "bias", None) is not None:
            state["bias_grad"] = module.bias.requires_grad
            module.bias.requires_grad_(False)
        bn_state.append(state)
    return bn_state


def _restore_bn_after_stage(bn_state):
    for state in bn_state:
        module = state["module"]
        module.train(state["training"])
        if getattr(module, "weight", None) is not None:
            module.weight.requires_grad_(state["weight_grad"])
        if getattr(module, "bias", None) is not None:
            module.bias.requires_grad_(state["bias_grad"])


def _find_classifier_modules(model):
    modules = []
    for attr_name in ("classifier", "fc", "head", "linear"):
        module = getattr(model, attr_name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    return modules


def _set_trainable_scope(model, head_only):
    if not head_only:
        for param in model.parameters():
            param.requires_grad_(True)
        return

    head_modules = _find_classifier_modules(model)
    if not head_modules:
        for param in model.parameters():
            param.requires_grad_(True)
        return

    trainable_ids = {id(param) for module in head_modules for param in module.parameters()}
    for param in model.parameters():
        param.requires_grad_(id(param) in trainable_ids)


def _find_last_linear(module):
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for submodule in reversed(list(module)):
            linear = _find_last_linear(submodule)
            if linear is not None:
                return linear
    return None


def _maybe_reset_forget_head_row(model, forget_loader, enabled):
    if not enabled or hasattr(model, "_ga_repair_v2_head_reset_done"):
        return

    dataset = forget_loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return

    unique_targets = np.unique(np.asarray(targets, dtype=np.int64))
    if len(unique_targets) != 1 or unique_targets[0] < 0:
        return

    forget_class = int(unique_targets[0])
    classifier = None
    for module in _find_classifier_modules(model):
        classifier = _find_last_linear(module)
        if classifier is not None:
            break
    if classifier is None or forget_class >= classifier.weight.size(0):
        return

    with torch.no_grad():
        classifier.weight[forget_class].zero_()
        if classifier.bias is not None:
            classifier.bias[forget_class].zero_()
    model._ga_repair_v2_head_reset_done = True


def _anchor_weight_for_epoch(args, epoch):
    start = float(getattr(args, "anchor_lambda_start", getattr(args, "anchor_lambda", 1e-4)))
    end = float(getattr(args, "anchor_lambda_end", start))
    total = max(1, int(getattr(args, "unlearn_epochs", 1)) - 1)
    frac = min(max(float(epoch) / total, 0.0), 1.0)
    return start + (end - start) * frac


def _build_teacher_from_mask(model, args, device):
    checkpoint = torch.load(args.mask, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    teacher = utils.build_model(args).to(device)
    current_mask = pruner.extract_mask(checkpoint)
    if current_mask:
        pruner.prune_model_custom(teacher, current_mask)
    teacher.load_state_dict(checkpoint, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@iterative_unlearn
def GA_repair_v2(data_loaders, model, criterion, optimizer, epoch, args):
    device = next(model.parameters()).device
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    utils.dataset_convert_to_train(retain_loader.dataset)
    utils.dataset_convert_to_train(forget_loader.dataset)

    if not hasattr(model, "_ga_repair_v2_teacher"):
        model._ga_repair_v2_teacher = _build_teacher_from_mask(model, args, device)
        model._ga_repair_v2_anchor_state = _make_anchor(model._ga_repair_v2_teacher)
        _maybe_reset_forget_head_row(model, forget_loader, getattr(args, "head_reset", False))

    teacher = model._ga_repair_v2_teacher
    anchor_state = model._ga_repair_v2_anchor_state
    anchor_weight = _anchor_weight_for_epoch(args, epoch)
    head_only = epoch < max(0, int(getattr(args, "forget_warmup_epochs", 0)))

    ga_opt = getattr(args, "_ga_opt_v2", None)
    if ga_opt is None:
        ga_opt = torch.optim.SGD(
            model.parameters(),
            lr=args.unlearn_lr,
            momentum=getattr(args, "momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 5e-4),
        )
        args._ga_opt_v2 = ga_opt

    repair_opt = getattr(args, "_repair_opt_v2", None)
    if repair_opt is None:
        repair_opt = SAM(
            model.parameters(),
            torch.optim.SGD,
            rho=getattr(args, "sam_rho", 2.0),
            adaptive=True,
            lr=getattr(args, "repair_lr", 0.01),
            momentum=getattr(args, "momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 5e-4),
        )
        args._repair_opt_v2 = repair_opt

    smoothing_eps = getattr(args, "smoothing_eps", 0.05)
    ce_ls = (
        LabelSmoothingCrossEntropy(eps=smoothing_eps)
        if smoothing_eps and smoothing_eps > 0
        else criterion
    )

    _set_trainable_scope(model, head_only=head_only)

    model.train()
    bn_state = _freeze_bn_for_stage(model) if getattr(args, "freeze_bn", False) else []

    forget_steps = max(1, int(getattr(args, "forget_steps", 1)))
    forget_weight = float(getattr(args, "forget_weight", 1.0))
    forget_ce_weight = float(getattr(args, "forget_ce_weight", 0.25))
    forget_margin = float(getattr(args, "forget_margin", 1.0))
    start = time.time()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    for batch_idx, (images, targets) in enumerate(forget_loader):
        if batch_idx >= forget_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        margin_loss = _forget_margin_loss(logits, targets, forget_margin)
        ce = F.cross_entropy(logits, targets)
        entropy = _entropy_mean(logits)
        anchor_loss = _l2_anchor_loss(model, anchor_state)

        loss = (
            forget_weight
            * (
                margin_loss
                - forget_ce_weight * ce
                - getattr(args, "entropy_beta", 0.5) * entropy
            )
            + anchor_weight * anchor_loss
        )

        if not (_tensor_is_finite(logits) and _tensor_is_finite(loss)):
            ga_opt.zero_grad(set_to_none=True)
            print(f"[GA_v2] skipped non-finite forget batch at epoch {epoch}, step {batch_idx}")
            continue

        ga_opt.zero_grad(set_to_none=True)
        loss.backward()
        if not _gradients_are_finite(model):
            ga_opt.zero_grad(set_to_none=True)
            print(f"[GA_v2] skipped non-finite forget gradients at epoch {epoch}, step {batch_idx}")
            continue
        _clip_trainable_grads(model, max_norm=5.0)
        ga_opt.step()

        with torch.no_grad():
            prec1 = utils.accuracy(logits.float().data, targets)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    if bn_state:
        _restore_bn_after_stage(bn_state)

    print(
        f"[GA_v2] epoch {epoch} steps {forget_steps}  "
        f"loss {losses.avg:.4f}  acc {top1.avg:.2f}  "
        f"anchor {anchor_weight:.6f}  head_only {head_only}  time {time.time()-start:.1f}s"
    )

    model.train()
    _set_trainable_scope(model, head_only=head_only)

    retain_steps = max(1, int(getattr(args, "retain_steps", 2)))
    retain_weight = float(getattr(args, "retain_weight", 1.0))
    kd_lambda = float(getattr(args, "kd_lambda", 1.0))
    kd_temperature = float(getattr(args, "distill_temperature", 2.0))
    mix_alpha = float(getattr(args, "mixup_alpha", 0.2))
    start = time.time()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    for batch_idx, (images, targets) in enumerate(retain_loader):
        if batch_idx >= retain_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mix_alpha > 0:
            lam = np.random.beta(mix_alpha, mix_alpha)
            index = torch.randperm(images.size(0), device=device)
            mixed_x = lam * images + (1.0 - lam) * images[index]
            y_a, y_b = targets, targets[index]
            with torch.no_grad():
                teacher_logits = teacher(mixed_x)

            logits = model(mixed_x)
            ce_loss = lam * ce_ls(logits, y_a) + (1.0 - lam) * ce_ls(logits, y_b)
            kd_loss = _distill_kl(logits, teacher_logits, kd_temperature)
            loss = retain_weight * (ce_loss + kd_lambda * kd_loss) + anchor_weight * _l2_anchor_loss(model, anchor_state)
            if not (_tensor_is_finite(logits) and _tensor_is_finite(teacher_logits) and _tensor_is_finite(loss)):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite mixed batch at epoch {epoch}, step {batch_idx}")
                continue
            repair_opt.zero_grad(set_to_none=True)
            loss.backward()
            if not _gradients_are_finite(model):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite first-step gradients at epoch {epoch}, step {batch_idx}")
                continue
            _clip_trainable_grads(model, max_norm=5.0)
            repair_opt.first_step(zero_grad=True)

            logits2 = model(mixed_x)
            ce_loss2 = lam * ce_ls(logits2, y_a) + (1.0 - lam) * ce_ls(logits2, y_b)
            kd_loss2 = _distill_kl(logits2, teacher_logits, kd_temperature)
            loss2 = retain_weight * (ce_loss2 + kd_lambda * kd_loss2) + anchor_weight * _l2_anchor_loss(model, anchor_state)
            if not (_tensor_is_finite(logits2) and _tensor_is_finite(loss2)):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite second mixed batch at epoch {epoch}, step {batch_idx}")
                continue
            loss2.backward()
            if not _gradients_are_finite(model):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite second-step gradients at epoch {epoch}, step {batch_idx}")
                continue
            _clip_trainable_grads(model, max_norm=5.0)
            repair_opt.second_step(zero_grad=True)

            with torch.no_grad():
                prec1 = utils.accuracy(logits2.float().data, y_a)[0]
                losses.update(loss2.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))
        else:
            with torch.no_grad():
                teacher_logits = teacher(images)

            logits = model(images)
            ce_loss = ce_ls(logits, targets)
            kd_loss = _distill_kl(logits, teacher_logits, kd_temperature)
            loss = retain_weight * (ce_loss + kd_lambda * kd_loss) + anchor_weight * _l2_anchor_loss(model, anchor_state)
            if not (_tensor_is_finite(logits) and _tensor_is_finite(teacher_logits) and _tensor_is_finite(loss)):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite retain batch at epoch {epoch}, step {batch_idx}")
                continue
            repair_opt.zero_grad(set_to_none=True)
            loss.backward()
            if not _gradients_are_finite(model):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite retain first-step gradients at epoch {epoch}, step {batch_idx}")
                continue
            _clip_trainable_grads(model, max_norm=5.0)
            repair_opt.first_step(zero_grad=True)

            logits2 = model(images)
            ce_loss2 = ce_ls(logits2, targets)
            kd_loss2 = _distill_kl(logits2, teacher_logits, kd_temperature)
            loss2 = retain_weight * (ce_loss2 + kd_lambda * kd_loss2) + anchor_weight * _l2_anchor_loss(model, anchor_state)
            if not (_tensor_is_finite(logits2) and _tensor_is_finite(loss2)):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite retain second batch at epoch {epoch}, step {batch_idx}")
                continue
            loss2.backward()
            if not _gradients_are_finite(model):
                repair_opt.zero_grad(set_to_none=True)
                print(f"[Repair_v2] skipped non-finite retain second-step gradients at epoch {epoch}, step {batch_idx}")
                continue
            _clip_trainable_grads(model, max_norm=5.0)
            repair_opt.second_step(zero_grad=True)

            with torch.no_grad():
                prec1 = utils.accuracy(logits2.float().data, targets)[0]
                losses.update(loss2.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))

    _set_trainable_scope(model, head_only=False)

    print(
        f"[Repair_v2] epoch {epoch} steps {retain_steps}  "
        f"loss {losses.avg:.4f}  acc {top1.avg:.2f}  "
        f"kd {kd_lambda:.3f}  time {time.time()-start:.1f}s"
    )
    return top1.avg
