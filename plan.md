# GA_Repair Improvement Plan

## Goal

Modify `GA_repair` so its classwise unlearning result moves as close as possible to the retraining gold standard on HAM10000 with `efficientnet_b0`.

Current target comparison:

- Dense retrain:
  - retain `95.39`
  - forget `0.00`
  - val `81.16`
  - test `81.97`
- Dense `GA_repair`:
  - retain `73.93`
  - forget `30.04`
  - val `70.74`
  - test `72.54`

- Sparse retrain (`p80`):
  - retain `88.20`
  - forget `0.00`
  - val `77.22`
  - test `78.13`
- Sparse `GA_repair`:
  - retain `76.18`
  - forget `10.30`
  - val `72.01`
  - test `74.21`

## Main Observation

The current `GA_repair` recipe is not failing in only one direction.

- On the dense branch, it is too destructive:
  - utility drops too much
  - forget accuracy is still far from retrain
- On the sparse branch, it is closer to retrain than dense, but still leaves a meaningful gap

So the next step is not just more hyperparameter search. We need a modified algorithm.

## Working Hypothesis

The current method is overusing early aggressive forget-side ascent and under-constraining retain-side recovery.

Likely failure modes:

1. Forget ascent damages shared features too early.
2. Repair is too weak and too short to recover retained behavior.
3. There is no explicit mechanism that says:
   - forget class `0`
   - while preserving the original function on non-forget data
4. The same recipe is being applied to dense and sparse models, even though they behave differently.

## Proposed Algorithm Changes

### 1. Add Retain-Side Distillation

Keep a frozen teacher copy of the pre-unlearning checkpoint and add a retain-side KL/logit distillation loss.

Reason:

- retraining keeps good retain/val/test because it relearns a clean retain-only solution
- `GA_repair` currently has no strong constraint to preserve the original decision function on retained classes
- distillation should reduce the large retain/test gap

Suggested form:

- On retain batches:
  - `L_retain = CE(retain) + lambda_kd * KL(student_logits, teacher_logits)`

Initial sweep:

- `lambda_kd in {0.5, 1.0, 2.0}`

### 2. Two-Stage Unlearning

Split `GA_repair` into:

1. short classifier-focused forget stage
2. retain-heavy repair stage

Reason:

- the dense run suggests full-network ascent harms shared representation too much
- classwise forgetting should first target the classifier head or top layers
- only then should the backbone be allowed to move

Suggested structure:

- Stage A:
  - 2-4 epochs
  - update classifier head only
  - strong forget objective
- Stage B:
  - 8-12 epochs
  - unfreeze full model
  - weaker forget objective
  - stronger retain repair + distillation

### 3. Replace Pure CE Ascent with Targeted Logit Suppression

Instead of relying mainly on `-CE(forget)`, directly push down the forgotten-class logit with a margin loss.

Reason:

- `-CE` often damages shared features broadly
- classwise forgetting needs targeted suppression of the forgotten class

Suggested form:

- `L_forget = max(0, z_y - max(z_other) + margin)`
- plus optional entropy/confusion term

Initial sweep:

- `margin in {0.5, 1.0, 1.5}`

### 4. Make Anchor Strength Scheduled, Not Fixed

Use a time-varying `anchor_lambda`.

Reason:

- early epochs need freedom to forget
- later epochs need stronger pull back toward useful retained behavior

Suggested schedule:

- epochs 0-3: low anchor
- epochs 4+: stronger anchor

Initial values:

- early `anchor_lambda`: `2e-5` to `5e-5`
- late `anchor_lambda`: `1e-4` to `3e-4`

### 5. Make Repair Stronger Than Forget on Dense Models

Do not use the sparse-tuned ratio on dense models.

Reason:

- dense `GA_repair` collapsed much harder than sparse
- dense branch needs more recovery pressure

Suggested dense defaults:

- `forget_steps = 1 or 2`
- `retain_steps = 3 or 4`
- lower `unlearn_lr`
- slightly higher `repair_lr`

Suggested sparse defaults:

- keep `forget_steps = 2 or 3`
- `retain_steps = 2`

### 6. Freeze BN During Forget Stage

Use `--freeze_bn` during ascent-heavy updates, then recalibrate BN at the end.

Reason:

- small forget batches can corrupt running statistics
- this is more likely in classwise forgetting

### 7. Add Early Stopping Using a Retrain-Gap Score

Track a scalar score during evaluation:

- `score = a * retain_gap + b * forget_gap + c * val_gap + d * test_gap`

with retrain as reference.

Reason:

- best unlearning epoch is not necessarily the last epoch
- current runs only report final values

Suggested priority:

- emphasize retain + forget first
- val/test second

## Concrete Version To Implement First

Implement `GA_repair_v2` with:

1. teacher distillation on retain batches
2. classifier-only forget warmup
3. logit-margin forget loss instead of only `-CE`
4. scheduled anchoring
5. BN freeze during forget stage

This is the highest-value first modification because it addresses both:

- dense utility collapse
- incomplete forgetting

## Experiment Order

### Phase 1: Dense Branch First

Use dense classwise forgetting as the primary development benchmark.

Reason:

- dense retrain gives the cleanest gold standard
- current dense `GA_repair` gap is the worst
- if dense cannot be improved, sparse will not be convincing

Target:

- retain `> 88`
- forget `< 10`
- val `> 78`
- test `> 79`

### Phase 2: Transfer to Sparse `p80`

After dense improves, test the same method on sparse `p80`.

Target:

- retain `> 84`
- forget `< 5`
- val `> 75`
- test `> 77`

## Immediate Implementation Tasks

1. Create `GA_repair_v2.py`
2. Add teacher-model loading from the original checkpoint
3. Add retain-side KL distillation
4. Add margin-based forget loss
5. Add stage-wise parameter freezing
6. Add scheduled anchor weight
7. Add per-epoch evaluation tracking against retrain reference
8. Compare:
   - current `GA_repair`
   - `GA_repair_v2` dense
   - `GA_repair_v2` sparse

## Success Criterion

We should claim progress only if `GA_repair_v2` reduces the gap to retrain on both:

- forgetting quality
- retained utility

and not just one of them.
