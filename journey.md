# GA_Repair Journey

## Scope

- Dataset: HAM10000
- Model: `efficientnet_b0`
- Starting checkpoint: `ham10000_effb0/0model_SA_best.pth.tar`
- Current focus: dense-model unlearning with `GA_repair`
- Classwise forgetting target: class `0`

## Best Result So Far

Best overall tradeoff so far is:

- Run: `ham10000_ga_repair_class0_fs3_rs2_lr4e3_a7e5`
- Unlearning method: `GA_repair`
- Forget setting: classwise forget, `--class_to_replace 0`

### Hyperparameters

```bash
python -u main_forget.py \
  --data ./data/ham10000 \
  --dataset ham10000 \
  --arch efficientnet_b0 \
  --input_size 224 \
  --num_classes 7 \
  --save_dir ham10000_ga_repair_class0_fs3_rs2_lr4e3_a7e5 \
  --mask ham10000_effb0/0model_SA_best.pth.tar \
  --unlearn GA_repair \
  --class_to_replace 0 \
  --unlearn_lr 0.004 \
  --repair_lr 0.005 \
  --retain_steps 2 \
  --forget_steps 3 \
  --entropy_beta 1.5 \
  --anchor_lambda 7e-5 \
  --smoothing_eps 0.05 \
  --mixup_alpha 0.2 \
  --sam_rho 2.0 \
  --unlearn_epochs 12 \
  --bn_recalibrate_batches 128 \
  --batch_size 64 \
  --workers 8
```

### Result

- retain acc: `90.20`
- forget acc: `18.88`
- val acc: `75.62`
- test acc: `78.48`

### Why this is the current best

- It forgets much better than the weak/balanced settings.
- It preserves utility much better than the most aggressive setting.
- Among the sweep runs, this is the best compromise between forgetting and retained performance.

## Completed Runs

### Initial classwise GA_Repair

- Run: `ham10000_ga_repair_class0`
- retain acc: `99.12`
- forget acc: `97.42`
- val acc: `78.69`
- test acc: `79.18`
- Conclusion: did not forget effectively.

### Correct random forgetting baseline

- Run: `ham10000_ga_repair_random233_v2`
- Forget count matched class-0 train size: `233`
- retain acc: `98.82`
- forget acc: `99.14`
- val acc: `77.82`
- test acc: `76.08`
- Conclusion: also too weak on forgetting.

### Stronger classwise GA_Repair

- Run: `ham10000_ga_repair_class0_stronger`
- retain acc: `74.20`
- forget acc: `1.72`
- val acc: `70.41`
- test acc: `72.68`
- Conclusion: forgetting worked, but utility collapsed too much.

### Balanced classwise GA_Repair

- Run: `ham10000_ga_repair_class0_balanced`
- retain acc: `96.87`
- forget acc: `86.70`
- val acc: `77.22`
- test acc: `79.59`
- Conclusion: utility recovered, but forgetting became too weak again.

### Sweep Run 1

- Run: `ham10000_ga_repair_class0_fs3_rs1_lr4e3_a7e5`
- retain acc: `85.14`
- forget acc: `17.17`
- val acc: `73.15`
- test acc: `76.59`
- Conclusion: strong forgetting, but weaker utility than the best run.

### Sweep Run 2

- Run: `ham10000_ga_repair_class0_fs3_rs2_lr4e3_a7e5`
- retain acc: `90.20`
- forget acc: `18.88`
- val acc: `75.62`
- test acc: `78.48`
- Conclusion: current best tradeoff.

### Sweep Run 3

- Run: `ham10000_ga_repair_class0_fs2_rs1_lr4e3_ent14`
- retain acc: `91.34`
- forget acc: `65.24`
- val acc: `74.95`
- test acc: `78.62`
- Conclusion: utility is acceptable, but forgetting is too weak.

## Notes

- The dense checkpoint is being used for all current `GA_repair` experiments.
- The `mask` warnings in logs are expected in this phase because the checkpoint is dense and the remain-weight ratio stays at `100%`.
- A previous bug made random forgetting silently behave like classwise forgetting because `--class_to_replace` defaulted to `0`; this was fixed by restoring the default to `-1`.

## Next GA_Repair Direction

If continuing the search, stay near the current best setting and search around:

- `forget_steps = 3`
- `retain_steps = 2`
- `unlearn_lr` between `0.004` and `0.0045`
- `anchor_lambda` between `6e-5` and `7e-5`

## GA_Repair_v2 Summary

Updated: `2026-04-12 13:47:55 +06`

### Method Changes

`GA_repair_v2` adds:

- retain-side teacher distillation
- classifier-only warmup
- margin-based forget loss
- scheduled anchoring
- BN freezing during forget updates
- sparse-stability guards:
  - non-finite loss/gradient checks
  - gradient clipping

### Best Thesis-Grade Result

Best overall result so far is:

- Run: `ham10000_ga_repair_v2_class0_dense_real`
- Setting: classwise forget, dense network, forget class `0`
- Method: `GA_repair_v2`

Result:

- retain acc: `94.52`
- forget acc: `0.00`
- val acc: `82.23`
- test acc: `82.53`
- runtime: `341.37` seconds (`5.69` minutes)

Gold-standard comparison:

- Retrain run: `ham10000_retrain_class0_dense`
- retrain retain acc: `95.39`
- retrain forget acc: `0.00`
- retrain val acc: `81.16`
- retrain test acc: `81.97`
- retrain runtime: about `3890.97` seconds (`64.85` minutes)

Conclusion:

- `GA_repair_v2` is essentially at dense retrain quality.
- It is about `11x` faster than dense retraining.

### Final Results Table

#### Classwise Forgetting

Dense `GA_repair_v2`

- Run: `ham10000_ga_repair_v2_class0_dense_real`
- retain acc: `94.52`
- forget acc: `0.00`
- val acc: `82.23`
- test acc: `82.53`
- runtime: `341.37` seconds

Dense retrain

- Run: `ham10000_retrain_class0_dense`
- retain acc: `95.39`
- forget acc: `0.00`
- val acc: `81.16`
- test acc: `81.97`
- runtime: `3890.97` seconds

Dense comparison

- retain gap to retrain: `-0.87`
- forget gap to retrain: `0.00`
- val gap to retrain: `+1.07`
- test gap to retrain: `+0.56`

Sparse `GA_repair_v2` initial run

- Run: `ham10000_ga_repair_v2_class0_sparse_p80_real`
- retain acc: `0.00`
- forget acc: `100.00`
- val acc: `3.01`
- test acc: `0.00`
- Status: invalid run due to numerical instability (`NaN`)

Sparse `GA_repair_v2` safe run

- Run: `ham10000_ga_repair_v2_class0_sparse_p80_safe`
- retain acc: `91.95`
- forget acc: `82.40`
- val acc: `81.83`
- test acc: `82.04`
- runtime: `221.37` seconds
- Status: stable but forgetting too weak

Sparse `GA_repair_v2` best valid run

- Run: `ham10000_ga_repair_v2_class0_sparse_p80_mid`
- retain acc: `91.09`
- forget acc: `2.58`
- val acc: `80.96`
- test acc: `82.81`
- runtime: `231.71` seconds

Sparse retrain

- Run: `ham10000_retrain_class0_sparse_p80`
- retain acc: `88.20`
- forget acc: `0.00`
- val acc: `77.22`
- test acc: `78.13`

Sparse comparison using best valid `GA_repair_v2`

- retain gap to retrain: `+2.89`
- forget gap to retrain: `+2.58`
- val gap to retrain: `+3.74`
- test gap to retrain: `+4.68`

Conclusion:

- Sparse `GA_repair_v2` now works well.
- It does not exactly match retrain on forgetting, but it is close enough to be credible.
- Utility is clearly stronger than sparse retrain.

#### Random Forgetting (`233` samples)

Dense `GA_repair_v2`

- Run: `ham10000_ga_repair_v2_random233_dense_real`
- retain acc: `94.89`
- forget acc: `94.42`
- val acc: `82.30`
- test acc: `82.03`
- runtime: `344.65` seconds

Dense retrain

- Run: `ham10000_retrain_random233_dense`
- retain acc: `94.25`
- forget acc: `89.70`
- val acc: `80.23`
- test acc: `78.58`
- runtime: `4166.73` seconds

Dense random comparison

- retain gap to retrain: `+0.63`
- forget gap to retrain: `+4.72`
- val gap to retrain: `+2.07`
- test gap to retrain: `+3.45`

Sparse `GA_repair_v2`

- Run: `ham10000_ga_repair_v2_random233_sparse_p80_real`
- retain acc: `91.55`
- forget acc: `91.42`
- val acc: `82.63`
- test acc: `81.28`
- runtime: `319.19` seconds

Sparse retrain

- Run: `ham10000_retrain_random233_sparse_p80`
- retain acc: `85.72`
- forget acc: `86.27`
- val acc: `75.89`
- test acc: `75.00`
- Note: only `train.log` exists for this older retrain run; no final checkpoint was saved

Sparse random comparison

- retain gap to retrain: `+5.83`
- forget gap to retrain: `+5.15`
- val gap to retrain: `+6.75`
- test gap to retrain: `+6.28`

Conclusion:

- On random forgetting, retrain itself still has high forget accuracy.
- So classwise forgetting remains the better benchmark for thesis claims.
- `GA_repair_v2` preserves utility very well on random forgetting for both dense and sparse models.

### MIA Notes

MIA was recorded for:

- `ham10000_ga_repair_v2_class0_dense_real`
- `ham10000_ga_repair_v2_class0_sparse_p80_mid`
- `ham10000_ga_repair_v2_random233_dense_real`
- `ham10000_ga_repair_v2_random233_sparse_p80_real`

High-level interpretation:

- training-privacy MIA stays in a similar band across retrain and `GA_repair_v2`
- the accuracy comparison is the stronger result
- classwise forgetting remains the cleanest evidence

### Current Takeaway

- Original `GA_repair` had a large gap to retrain.
- `GA_repair_v2` closes that gap substantially.
- Dense classwise forgetting is now near-retrain quality.
- Sparse classwise forgetting is also strong after stability fixes.
- `GA_repair_v2` is the version to report going forward, not the original `GA_repair`.
