# W1 Run Checkpoint (HumanoidRun, walk-only)

| Field | Value |
|---|---|
| Date | 2026-05-06 |
| Source script | `tokenhsi/scripts/single_task/run_train.sh` |
| Task class | `HumanoidRun` (commit f43bc87, fix 2e871ef) |
| Env config | `amp_humanoid_run.yaml` (W1: speedMin=0.5, speedMax=1.5) |
| Motion data | `dataset_run.yaml` → 12 walk segments (`dataset_amass_loco/motions/`) |
| num_envs | 1024 |
| Total epochs | 3308 |
| Wall clock | 1h 52min |
| Hardware | RTX 2080 Ti (single GPU) |
| Final fps_total | ~16,000 |

## Convergence (TensorBoard)

- Training mean reward (last 20 iter): **154.16 ± 2.09** (random target switches every 0.5% steps)
- AMP disc loss: 20.34 → 0.38 (healthy)
- AMP disc agent acc: 54% → 99%
- AMP disc demo acc: 83% → 100%

## Evaluation (94 random eval episodes)

- All episodes ran to full 299 steps (**100% no-fall rate**)
- Episode reward: **203.4 ± 21.0** (range [134.75, 245.04])
- Estimated mean tracking error: **≈ 0.19 m/s** (success threshold 0.3 m/s)

## Artifacts

- Checkpoint: `output/single_task/ckpt_run_w1.pth` (40 MB)
- Random seed: `output/single_task/ckpt_run_w1.seed.txt`
- Training run dir: `output/Humanoid_06-13-14-57/` (TensorBoard summaries)
- Eval log: `/tmp/run_w1_eval.log`
