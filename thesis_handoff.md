# Thesis Handoff

## Core Thesis Idea

The thesis goal is to improve approximate machine unlearning for sparse and dense neural
networks, with a focus on making `GA_repair` behave closer to the gold-standard
retraining baseline.

The main research question became:

- Can we modify gradient-ascent-plus-repair unlearning so that it forgets target data
  effectively while preserving retained utility, especially under sparsity?

The practical benchmark used in the later stage of the work is:

- dataset: `HAM10000`
- model: `efficientnet_b0`
- forgetting settings:
  - classwise forgetting: forget class `0`
  - random forgetting: forget `233` samples
- networks:
  - dense
  - sparse `p80` (`20%` remaining weights)


## Why The Work Shifted

Initial work on the original `GA_repair` showed a consistent tradeoff problem:

- weak settings did not forget enough
- strong settings forgot well but damaged retain/test performance
- sparse settings were even harder to balance

At the same time, some baseline issues had to be fixed first:

1. The original HAM10000 EfficientNet-B0 training recipe was overfitting.
2. Extremely aggressive pruning (`rate=0.95`) was too destructive and caused severe
   underfitting in the sparse branch.
3. A class-balanced-loss bug was making retrain baselines artificially weak in
   retain-only settings.

So the workflow became:

1. fix baseline training
2. fix retraining correctness
3. identify valid dense and sparse baselines
4. redesign `GA_repair`


## Baseline Training Changes

The HAM10000 baseline was improved before final unlearning experiments.

Main changes:

- stronger HAM10000 augmentation
- lower learning rate
- fewer training epochs
- class-balanced loss
- label smoothing
- proper sparse checkpoint generation with `pruning_times=2`

Most important baseline branch used later:

- `ham10000_effb0_baseline_v2_p80`

Important artifacts from that branch:

- dense best checkpoint:
  - `ham10000_effb0_baseline_v2_p80/0model_SA_best.pth.tar`
- sparse best checkpoint:
  - `ham10000_effb0_baseline_v2_p80/1model_SA_best.pth.tar`
- rewind weights:
  - `ham10000_effb0_baseline_v2_p80/epoch_8_rewind_weight.pt`


## Why Sparse `p80` Was Chosen

The first sparse branch used very aggressive sparsity and was too damaging.
That did not produce a fair unlearning benchmark.

The later decision was to use:

- `rate=0.8`
- `pruning_times=2`

This gives a sparse model with about `20%` remaining weights, which was much more
usable than the earlier highly sparse branch.


## Original GA_repair Findings

Before `GA_repair_v2`, multiple `GA_repair` sweeps were run.

Main conclusion from the original method:

- it could either forget or preserve utility, but usually not both well enough
- the gap to retraining was still large
- dense classwise forgetting was the cleanest benchmark showing the weakness

The best old `GA_repair` setting was:

- `forget_steps=3`
- `retain_steps=2`
- `unlearn_lr=0.004`
- `repair_lr=0.005`
- `anchor_lambda=7e-5`

But even that was still not close enough to retraining.


## GA_repair_v2 Idea

`GA_repair_v2` was introduced as a controlled upgrade of `GA_repair`.

The design principle was:

- make forgetting more targeted
- make repair more preservation-oriented
- make sparse optimization stable enough to evaluate seriously

Main changes introduced in `GA_repair_v2`:

1. retain-side teacher distillation
2. classifier-only warmup in early epochs
3. margin-based forget loss
4. scheduled anchoring
5. BN freezing during forget updates
6. optional forgotten-class head reset
7. sparse-safe teacher construction
8. numerical stability guards:
   - finite-value checks
   - finite-gradient checks
   - gradient clipping

Main implementation files:

- `unlearn/GA_repair_v2.py`
- `unlearn/__init__.py`
- `arg_parser.py`

Supporting fixes:

- `utils.py`
- `main_forget.py`
- `unlearn/impl.py`


## Important Bug Fixes Along The Way

Several non-algorithmic issues materially affected results:

1. `rewind_lt` crash in `main_imp.py`
- fixed fallback behavior when no rewind snapshot was available

2. random forgetting bug
- `--class_to_replace` had been defaulting incorrectly in one phase
- corrected so random forgetting actually used `--class_to_replace -1`

3. class-balanced-loss bug in retain-only retraining
- absent classes were collapsing useful class weights
- fixed in `utils.py`

4. retrain resume/checkpointing
- per-epoch progress checkpoints were added for `main_forget.py`

5. sparse teacher failure in `GA_repair_v2`
- `copy.deepcopy(model)` failed on masked sparse models
- replaced with fresh model reconstruction + mask loading

6. sparse numerical instability in `GA_repair_v2`
- strong sparse classwise settings produced `NaN`s
- fixed by adding stability guards and then retuning the sparse hyperparameters


## Final Experimental Findings

### 1. Dense Classwise Forgetting

This is the strongest result of the thesis work.

Dense `GA_repair_v2`:

- run: `ham10000_ga_repair_v2_class0_dense_real`
- retain: `94.52`
- forget: `0.00`
- val: `82.23`
- test: `82.53`
- runtime: `341.37` seconds

Dense retrain:

- run: `ham10000_retrain_class0_dense`
- retain: `95.39`
- forget: `0.00`
- val: `81.16`
- test: `81.97`
- runtime: `3890.97` seconds

Conclusion:

- `GA_repair_v2` is essentially at dense retrain quality
- it is about `11x` faster than retraining


### 2. Sparse Classwise Forgetting

There were three phases:

- first sparse `GA_repair_v2` run: invalid due to instability
- second sparse run: stable but too weak on forgetting
- third sparse run: stable and strong

Best valid sparse `GA_repair_v2`:

- run: `ham10000_ga_repair_v2_class0_sparse_p80_mid`
- retain: `91.09`
- forget: `2.58`
- val: `80.96`
- test: `82.81`
- runtime: `231.71` seconds

Sparse retrain:

- run: `ham10000_retrain_class0_sparse_p80`
- retain: `88.20`
- forget: `0.00`
- val: `77.22`
- test: `78.13`

Conclusion:

- sparse `GA_repair_v2` does not exactly match retrain on forget accuracy
- but it is close enough to be credible
- utility is clearly stronger than sparse retrain
- this is a strong approximate-unlearning result


### 3. Dense Random Forgetting

Dense `GA_repair_v2`:

- run: `ham10000_ga_repair_v2_random233_dense_real`
- retain: `94.89`
- forget: `94.42`
- val: `82.30`
- test: `82.03`
- runtime: `344.65` seconds

Dense retrain:

- run: `ham10000_retrain_random233_dense`
- retain: `94.25`
- forget: `89.70`
- val: `80.23`
- test: `78.58`

Conclusion:

- utility is better than retrain
- forgetting is slightly weaker than retrain
- but random-forgetting retrain itself still has high forget accuracy
- so classwise forgetting remains the cleaner benchmark


### 4. Sparse Random Forgetting

Sparse `GA_repair_v2`:

- run: `ham10000_ga_repair_v2_random233_sparse_p80_real`
- retain: `91.55`
- forget: `91.42`
- val: `82.63`
- test: `81.28`
- runtime: `319.19` seconds

Sparse retrain:

- run: `ham10000_retrain_random233_sparse_p80`
- only `train.log` exists, but the recorded metrics are:
  - retain: `85.72`
  - forget: `86.27`
  - val: `75.89`
  - test: `75.00`

Conclusion:

- again, utility is much stronger than retrain
- forgetting is slightly weaker
- random forgetting is not the main thesis evidence


## Main Thesis Takeaway

The strongest conclusion is:

- the original `GA_repair` had a large gap to retraining
- `GA_repair_v2` closes that gap substantially
- on dense classwise forgetting, the gap is almost gone
- on sparse classwise forgetting, the method is also strong after stability fixes

So the thesis claim should center on:

- `GA_repair_v2` as a more controlled and effective approximate unlearning method
- especially for classwise forgetting
- with strong utility preservation
- and much lower runtime than retraining


## Best Runs To Preserve

If moving to another device, the most important result directories are:

- `ham10000_effb0_baseline_v2_p80`
- `ham10000_ga_repair_v2_class0_dense_real`
- `ham10000_ga_repair_v2_class0_sparse_p80_mid`
- `ham10000_ga_repair_v2_random233_dense_real`
- `ham10000_ga_repair_v2_random233_sparse_p80_real`
- `ham10000_retrain_class0_dense`
- `ham10000_retrain_class0_sparse_p80`
- `ham10000_retrain_random233_dense`
- `ham10000_retrain_random233_sparse_p80`


## Best Final Comparison To Report

If only one dense and one sparse result should be highlighted, use:

Dense classwise:

- `GA_repair_v2`: `94.52 / 0.00 / 82.23 / 82.53`
- retrain: `95.39 / 0.00 / 81.16 / 81.97`

Sparse classwise:

- `GA_repair_v2`: `91.09 / 2.58 / 80.96 / 82.81`
- retrain: `88.20 / 0.00 / 77.22 / 78.13`

Order of metrics:

- retain / forget / val / test


## Current Code State

The current codebase already contains:

- the baseline training fixes
- retrain fixes
- resume/progress checkpoint support
- `GA_repair_v2`
- sparse stability guards

So if the project is moved to another machine, the main task is not to re-implement
the method. The main task is to preserve the important checkpoints and rerun only if
new experiments are needed.
