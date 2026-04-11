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

