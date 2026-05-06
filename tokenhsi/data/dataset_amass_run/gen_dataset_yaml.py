"""Auto-generate ``dataset_amass_run.yaml`` by scanning ``motions/`` directory.

Run this AFTER ``preprocess.py`` and ``generate_motion.py`` have produced
the retargeted ``ref_motion.npy`` files.

Usage:
    python tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py

Writes ``dataset_amass_run.yaml`` listing all retargeted motions under
the ``loco_run`` skill key, each with weight 1.0.
"""
import os
import sys
import glob
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIONS_DIR = os.path.join(THIS_DIR, "motions")
OUTPUT_YAML = os.path.join(THIS_DIR, "dataset_amass_run.yaml")
SKILL_KEY = "loco_run"   # must match amp_humanoid_run.yaml's skill field (W2)


def main():
    if not os.path.isdir(MOTIONS_DIR):
        print(f"[ERROR] motions/ 目录不存在: {MOTIONS_DIR}", file=sys.stderr)
        print("       请先运行 generate_motion.py 产出 retarget 后的 ref_motion.npy", file=sys.stderr)
        sys.exit(1)

    pattern = os.path.join(MOTIONS_DIR, "*", "*", "phys_humanoid_v3", "ref_motion.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] 未找到任何 ref_motion.npy 文件 (pattern: {pattern})", file=sys.stderr)
        print("       请先运行 generate_motion.py", file=sys.stderr)
        sys.exit(1)

    entries = []
    for f in files:
        rel = os.path.relpath(f, THIS_DIR)
        entries.append({"file": rel, "weight": 1.0})

    cfg = {"motions": {SKILL_KEY: entries}}
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {len(entries)} motions to {OUTPUT_YAML}")
    for e in entries:
        print(f"  - {e['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
