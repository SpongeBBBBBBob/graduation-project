"""Verify AMASS file existence for the `run` motion list.

Usage:
    python tokenhsi/data/dataset_amass_run/check_amass_files.py

Reads ``dataset_cfg.yaml``'s ``amass_dir`` and ``motions.run`` list,
checks each candidate file's existence under ``<amass_dir>``, and writes
two report files in the same directory:

  - ``available_files.txt``  -- candidates that actually exist
  - ``missing_files.txt``    -- candidates that are missing

Exit code 1 on configuration errors (placeholder ``amass_dir``,
missing ``motions.run`` section, or non-existent ``amass_dir``).
"""
import os
import sys
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CFG = os.path.normpath(os.path.join(THIS_DIR, "..", "dataset_cfg.yaml"))
OUTPUT_AVAILABLE = os.path.join(THIS_DIR, "available_files.txt")
OUTPUT_MISSING = os.path.join(THIS_DIR, "missing_files.txt")


def main():
    with open(DATASET_CFG, "r") as f:
        cfg = yaml.safe_load(f)

    amass_dir = cfg.get("amass_dir", "")
    if not amass_dir or amass_dir.startswith("/YOUR_PATH"):
        print("[ERROR] dataset_cfg.yaml 中 amass_dir 仍为占位符或为空，请先填真实路径", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(amass_dir):
        print(f"[ERROR] amass_dir 不存在: {amass_dir}", file=sys.stderr)
        sys.exit(1)
    if "run" not in cfg.get("motions", {}):
        print("[ERROR] dataset_cfg.yaml 中没有 motions.run 节", file=sys.stderr)
        sys.exit(1)

    candidates = cfg["motions"]["run"]
    available, missing = [], []
    for seq in candidates:
        rel_path = seq.replace("+__+", "/")
        full_path = os.path.join(amass_dir, rel_path)
        if os.path.isfile(full_path):
            available.append(seq)
            print(f"[OK]    {seq}")
        else:
            missing.append(seq)
            print(f"[MISS]  {seq}")

    with open(OUTPUT_AVAILABLE, "w") as f:
        for s in available:
            f.write(s + "\n")
    with open(OUTPUT_MISSING, "w") as f:
        for s in missing:
            f.write(s + "\n")

    print()
    print(f"Total: {len(candidates)}  Available: {len(available)}  Missing: {len(missing)}")
    print(f"Available list: {OUTPUT_AVAILABLE}")
    print(f"Missing list:   {OUTPUT_MISSING}")

    if len(available) < 6:
        print()
        print("[WARN] 可用片段 < 6 段，可能不足以训练 jog/run。")
        print("       建议通过关键字扫描扩充候选清单（grep AMASS 中含 run/jog/sprint 的文件名）。")

    return 0 if len(available) >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
