#!/bin/bash
# 从 output/train_stage1/nn/Humanoid.pth 继续训练
# 仅保持 random_seed 与原训练一致

SEED_ARG=""
if [ -f "output/train_stage1/random_seed.txt" ]; then
    # random_seed.txt 可能是科学计数法（如 2.326e+03），需要正确解析成整数 2326
    SEED=$(python - <<'PY'
import re

with open("output/train_stage1/random_seed.txt", "r", encoding="utf-8") as f:
    line = f.readline().strip()
match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
if not match:
    raise SystemExit(1)
print(int(float(match.group(0))))
PY
)
    SEED_ARG="--seed $SEED"
    echo "Using seed from previous run: $SEED"
fi

horovodrun -np 10 python ./tokenhsi/run.py --task HumanoidTrajSitCarryClimb \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs 512 \
    --checkpoint output/train_stage1/nn/Humanoid.pth \
    --resume 1 \
    --output_path output/train_stage1 \
    $SEED_ARG \
    --horovod \
    --headless
