#!/bin/bash
# 通过 X11 转发运行，保存帧到本地，再转为 GIF/MP4
# 需先配置 X11 转发并启动 X410

set -e
cd "$(dirname "$0")/../../.."
OUTPUT_BASE="output"
MAX_FRAMES=${MAX_FRAMES:-300}

echo "Running traj_test with save_frames (max $MAX_FRAMES frames)..."
# checkpoint path
CHECKPOINT="output/Humanoid_22-09-32-45/nn/Humanoid.pth" #CHECKPOINT = "output/single_task/ckpt_traj.pth"
python ./tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml \
    --checkpoint $CHECKPOINT \
    --test \
    --num_envs 16 \
    --save_frames \
    --max_frames $MAX_FRAMES \
    --output_path $OUTPUT_BASE

OUTPUT_DIR=$(ls -td ${OUTPUT_BASE}/imgs/*/ 2>/dev/null | head -1)
FRAMES_DIR="${OUTPUT_DIR}frames"
if [ -z "$OUTPUT_DIR" ] || [ ! -d "$FRAMES_DIR" ]; then
    echo "No frames directory found."
    exit 1
fi

echo "Frames saved to: $FRAMES_DIR"
echo "Converting to GIF..."
python lpanlib/others/gif.py --imgs_dir "$FRAMES_DIR" --output_dir "$OUTPUT_DIR" --fps 60 --source_fps 60 --scale 0.5

echo "Converting to MP4..."
python lpanlib/others/video.py --imgs_dir "$FRAMES_DIR" --output_dir "$OUTPUT_DIR" --fps 60

echo "Done. Output:"
echo "  GIF: ${OUTPUT_DIR}video.gif"
echo "  MP4: ${OUTPUT_DIR}video.mp4"
