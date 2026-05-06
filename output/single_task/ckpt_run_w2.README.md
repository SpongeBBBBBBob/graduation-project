# W2 Run Checkpoint (HumanoidRun, walk + jog/run mixed)

| Field | Value |
|---|---|
| Date | 2026-05-06 |
| Source script | `tokenhsi/scripts/single_task/run_train.sh` |
| Task class | `HumanoidRun` |
| Env config | `amp_humanoid_run.yaml` (W2: speedMin=0.5, speedMax=3.5, skill=loco_run) |
| Motion data | `dataset_run.yaml` → 12 walk (weight 1.0) + 18 jog/run (weight 1.5) = 30 segments, 198.3s total |
| AMASS subsets used | ACCAD/Female1Running, ACCAD/Male2Running, CMU/09, KIT/3, HumanEva |
| num_envs | 1024 |
| Total epochs | 4887 |
| Wall clock | 2h 52min |
| Hardware | RTX 2080 Ti (single GPU) |
| Final fps_total | ~15,000 |

## Convergence (TensorBoard)

- Training mean reward (last 100 iter): **104 ± 2.8** (random target switches every 0.5% steps; 0.5–3.5 m/s range)
- AMP disc loss: 1.0 (healthy plateau)
- AMP disc agent acc: 99%
- Training reward plateau confirmed: first/second-half last-100-window means: 104.007 vs 104.033

## Evaluation (76 random eval episodes, full speed range 0.5–3.5 m/s)

- 71/76 = **93.4% no-fall rate** (vs W1's 100% on narrower 0.5–1.5 m/s range)
- Episode reward (full eps): **132.04 ± 45.12** (range [-1.4, 208.0])
- Estimated mean tracking error: **≈ 0.41 m/s**
- 5/76 episodes fell (high target_speed cases hardest); steps distribution 22–179

## W1 vs W2 Comparison

| Metric | W1 (speedMax=1.5) | W2 (speedMax=3.5) |
|---|---|---|
| No-fall rate | 100% | 93.4% |
| Mean reward | 203 ± 21 | 132 ± 45 |
| Tracking error | 0.19 m/s | 0.41 m/s |

W2 trades ~7% fall rate for 2.3× wider speed range (sprint capability).

## Artifacts

- Checkpoint: `output/single_task/ckpt_run_w2.pth` (40 MB)
- Random seed: `output/single_task/ckpt_run_w2.seed.txt`
- Training run dir: `output/Humanoid_06-15-32-23/` (TensorBoard summaries)
- Eval log: `output/single_task/ckpt_run_w2.eval.log`
