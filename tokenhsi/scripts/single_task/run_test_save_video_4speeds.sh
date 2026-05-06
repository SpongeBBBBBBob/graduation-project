#!/bin/bash
# 在 4 个固定 target speed 下分别录制 W2 ckpt 的视频，每段产出 mp4 + gif
# 需先配置 X11 转发并启动 X410（与 run_test_save_video.sh 一致）
# 总耗时：每段约 6-8 分钟（300 帧 + 渲染），4 段 ≈ 30 分钟

set -e
cd "$(dirname "$0")/../../.."

OUTPUT_BASE="output"
MAX_FRAMES=${MAX_FRAMES:-300}
CHECKPOINT=${CHECKPOINT:-"output/single_task/ckpt_run_w2.pth"}
VIDEO_DIR="output/run_videos"

mkdir -p "$VIDEO_DIR"

# 4 个固定速度（对应 amp_humanoid_run_video_s{10,18,25,35}.yaml）
declare -a SPEED_TAGS=("s10" "s18" "s25" "s35")
declare -a SPEED_VALS=("1.0" "1.8" "2.5" "3.5")
declare -a SPEED_LABELS=("walk" "fastwalk_jog" "jog" "sprint")

for i in "${!SPEED_TAGS[@]}"; do
    TAG="${SPEED_TAGS[$i]}"
    SPEED="${SPEED_VALS[$i]}"
    LABEL="${SPEED_LABELS[$i]}"
    CFG="tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run_video_${TAG}.yaml"

    echo ""
    echo "================================================================"
    echo "  [$((i+1))/4]  Speed ${SPEED} m/s  (${LABEL})"
    echo "================================================================"

    # 跑仿真 + 截屏
    python ./tokenhsi/run.py --task HumanoidRun \
        --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
        --cfg_env "$CFG" \
        --motion_file tokenhsi/data/dataset_run/dataset_run.yaml \
        --checkpoint "$CHECKPOINT" \
        --test \
        --num_envs 4 \
        --save_frames \
        --max_frames "$MAX_FRAMES" \
        --output_path "$OUTPUT_BASE"

    # 找最新的 imgs 目录
    OUTPUT_DIR=$(ls -td ${OUTPUT_BASE}/imgs/*/ 2>/dev/null | head -1)
    FRAMES_DIR="${OUTPUT_DIR}frames"
    if [ -z "$OUTPUT_DIR" ] || [ ! -d "$FRAMES_DIR" ]; then
        echo "[ERROR] No frames directory found for speed ${SPEED}; skip."
        continue
    fi

    echo "Frames saved to: $FRAMES_DIR"

    echo "Converting to GIF..."
    python lpanlib/others/gif.py --imgs_dir "$FRAMES_DIR" --output_dir "$OUTPUT_DIR" --fps 60 --source_fps 60 --scale 0.5

    echo "Converting to MP4..."
    python lpanlib/others/video.py --imgs_dir "$FRAMES_DIR" --output_dir "$OUTPUT_DIR" --fps 60

    # 归档到稳定名字
    if [ -f "${OUTPUT_DIR}video.mp4" ]; then
        cp "${OUTPUT_DIR}video.mp4" "${VIDEO_DIR}/run_w2_speed_${SPEED}_${LABEL}.mp4"
        echo "  -> ${VIDEO_DIR}/run_w2_speed_${SPEED}_${LABEL}.mp4"
    fi
    if [ -f "${OUTPUT_DIR}video.gif" ]; then
        cp "${OUTPUT_DIR}video.gif" "${VIDEO_DIR}/run_w2_speed_${SPEED}_${LABEL}.gif"
        echo "  -> ${VIDEO_DIR}/run_w2_speed_${SPEED}_${LABEL}.gif"
    fi
done

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "================================================================"
ls -lh "${VIDEO_DIR}/"
