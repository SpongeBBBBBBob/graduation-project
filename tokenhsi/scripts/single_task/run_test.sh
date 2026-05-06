#!/bin/bash

python ./tokenhsi/run.py --task HumanoidRun \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml \
    --motion_file tokenhsi/data/dataset_run/dataset_run.yaml \
    --checkpoint output/single_task/ckpt_run.pth \
    --test \
    --num_envs 16
