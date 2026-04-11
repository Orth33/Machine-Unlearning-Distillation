
import time
import torch
import torch.nn.functional as F

import utils
from LS import LabelSmoothingCrossEntropy
from .impl import iterative_unlearn

# Re-use anchor helpers from GA_repair to avoid duplication
try:
    from .GA_repair import _make_anchor, _l2_to_anchor
except Exception:
    _make_anchor = None
    _l2_to_anchor = None

def _complementary_ce(logits, targets, temperature=1.0, eps=1e-6):
    """Cross-entropy to a uniform distribution over all classes except the true label.
    Encourages the model to be maximally *wrong* on the given targets.
    Equivalent to maximizing CE w.r.t. the true label but numerically more stable.
    """
    C = logits.size(1)
    # Build target distribution q where q[y] = 0 and others = 1/(C-1)
    with torch.no_grad():
        q = torch.full((targets.size(0), C), fill_value=1.0 / (C - 1), device=logits.device)
        q.scatter_(1, targets.view(-1, 1), 0.0)
        # Avoid exact zeros for KL stability, then re-normalize
        q = q.clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)

    logp = F.log_softmax(logits / temperature, dim=1)
    loss = F.kl_div(logp, q, reduction='batchmean') * (temperature ** 2)
    return loss

def _entropy_maximization(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    # Maximize entropy -> minimize negative entropy
    ent = -(p * logp).sum(dim=1).mean()
    return -ent  # return as a loss to MINIMIZE (i.e., -entropy)

def _margin_pushdown(logits, targets, margin=1.0):
    """Enforce that the forgotten-class logit stays at least 'margin' below the best alternative."""
    y_logit = logits.gather(1, targets.view(-1, 1))
    masked = logits.clone()
    masked.scatter_(1, targets.view(-1, 1), float('-inf'))
    best_other = masked.max(dim=1, keepdim=True)[0]
    # We want: best_other - y_logit >= margin  ->  margin + y_logit - best_other <= 0
    return F.relu(margin + y_logit - best_other).mean()

def _maybe_make_anchor(model, cache_dict):
    if cache_dict.get('anchor', None) is None:
        if _make_anchor is not None:
            cache_dict['anchor'] = _make_anchor(model)
        else:
            # Fallback: simple snapshot
            cache_dict['anchor'] = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    return cache_dict['anchor']

def _l2_to_anchor_fallback(model, anchor):
    # Average L2 distance per-parameter (normalized by parameter count)
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    denom = 0
    for n, p in model.named_parameters():
        if (not p.requires_grad) or (n not in anchor):
            continue
        a = anchor[n]
        loss = loss + (p - a).pow(2).sum()
        denom += p.numel()
    if denom == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return loss / denom

def _l2_to_anchor_dispatch(model, anchor):
    if _l2_to_anchor is not None:
        return _l2_to_anchor(model, anchor)
    else:
        return _l2_to_anchor_fallback(model, anchor)

def _step_with_optimizer(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@iterative_unlearn
def FT_conf(data_loaders, model, criterion, optimizer, epoch, args):
    """Fine-tune with *retain-side repair* while actively confusing the model on the forget set.

    Within each epoch, we interleave:
      1) a *forget* step that pushes predictions for forgotten examples toward
         a uniform distribution over the non-target classes and enforces a margin
         against the true class (reduces Forget Acc and raises MIA efficacy),
      2) a *retain* step with label smoothing + L2 anchoring to the pre-unlearning
         weights to preserve retained accuracy.

    This is mask-aware by construction because the model is already pruned and
    torch.nn.utils.prune keeps masked weights inactive in the forward pass.
    """
    retain_loader = data_loaders.get('retain')
    forget_loader = data_loaders.get('forget')

    # Cache anchor once (epoch 0)
    cache = getattr(model, '_ft_conf_cache', {})
    anchor = _maybe_make_anchor(model, cache)
    model._ft_conf_cache = cache  # attach for subsequent epochs

    # Hyperparameters (with sensible defaults pulled from args if present)
    smoothing_eps = getattr(args, 'smoothing_eps', 0.05)
    anchor_lambda = getattr(args, 'anchor_lambda', 1e-4)
    entropy_beta = getattr(args, 'entropy_beta', 0.5)
    temp = getattr(args, 'forget_temperature', 2.0)
    margin = getattr(args, 'forget_margin', 1.0)
    forget_weight = getattr(args, 'forget_weight', 1.0)      # scales forget-side loss
    retain_weight = getattr(args, 'retain_weight', 1.0)      # scales retain-side loss
    steps_per_retain = getattr(args, 'steps_per_retain', 2)  # interleave ratio: 1 forget step then N retain steps

    # Switch to train mode
    model.train()
    start = time.time()

    # Build criterions
    retain_criterion = LabelSmoothingCrossEntropy(eps=smoothing_eps)

    # Iterators
    forget_iter = iter(forget_loader) if forget_loader is not None else None
    retain_iter = iter(retain_loader)

    # Number of inner steps per epoch
    total_steps = max(len(retain_loader), len(forget_loader) if forget_iter is not None else 0)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    device = next(model.parameters()).device

    step = 0
    while step < total_steps:
        # 1) One forget step (if we still have forget data)
        if forget_iter is not None:
            try:
                f_images, f_targets = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                f_images, f_targets = next(forget_iter)
            f_images = f_images.to(device, non_blocking=True)
            f_targets = f_targets.to(device, non_blocking=True)
            # labels may be negated in the loader to mark forgotten data; restore the real class ids
            # Original code often encodes forget labels as -y-1 -> recover as y = -label-1 when label<0
            if (f_targets < 0).any():
                real_targets = -f_targets - 1
            else:
                real_targets = f_targets

            logits_f = model(f_images)
            loss_forget = _complementary_ce(logits_f, real_targets, temperature=temp)
            loss_forget = loss_forget + entropy_beta * _entropy_maximization(logits_f)
            loss_forget = loss_forget + _margin_pushdown(logits_f, real_targets, margin=margin)
            _step_with_optimizer(forget_weight * loss_forget, optimizer)

        # 2) K retain steps
        for _ in range(steps_per_retain):
            try:
                r_images, r_targets = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_images, r_targets = next(retain_iter)
            r_images = r_images.to(device, non_blocking=True)
            r_targets = r_targets.to(device, non_blocking=True)

            logits_r = model(r_images)
            loss_retain = retain_criterion(logits_r, r_targets)
            # L2 anchoring to pre-unlearning weights to preserve sparsified solution
            loss_retain = loss_retain + anchor_lambda * _l2_to_anchor_dispatch(model, anchor)

            _step_with_optimizer(retain_weight * loss_retain, optimizer)

            # Track train acc on retain batches
            with torch.no_grad():
                prec1 = utils.accuracy(logits_r, r_targets)[0]
                top1.update(prec1.item(), r_images.size(0))
                losses.update(loss_retain.item(), r_images.size(0))

        step += 1

    if (epoch == args.unlearn_epochs - 1) and hasattr(args, 'bn_recalibrate_batches'):
        # Optional: quick BN recalibration using retain data with train-time transforms
        try:
            utils.dataset_convert_to_train(retain_loader.dataset)
            model.train()
            with torch.no_grad():
                seen = 0
                for images, _ in retain_loader:
                    images = images.to(device, non_blocking=True)
                    _ = model(images)
                    seen += 1
                    if seen >= args.bn_recalibrate_batches:
                        break
            utils.dataset_convert_to_test(retain_loader.dataset, args)
            model.eval()
            print(f"[BN] Recalibrated BN with {seen} retain batches.")
        except Exception as e:
            print(f"[BN] Recalibration skipped due to error: {e}")

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg
