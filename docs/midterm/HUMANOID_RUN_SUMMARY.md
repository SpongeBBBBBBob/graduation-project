# 第五基础技能 HumanoidRun · 实验总结

- **作者**：毕设作者（与 AI 协助实施）
- **日期**：2026-05-05 ～ 2026-05-06
- **工程入口**：`tokenhsi/scripts/single_task/run_train.sh`
- **任务类**：`HumanoidRun`（位于 `tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py`）
- **设计文档**：[`docs/superpowers/specs/2026-05-05-humanoid-run-design.md`](../superpowers/specs/2026-05-05-humanoid-run-design.md)
- **实施计划**：[`docs/superpowers/plans/2026-05-05-humanoid-run.md`](../superpowers/plans/2026-05-05-humanoid-run.md)

---

## 1. 概览

在 TokenHSI 现有 4 个基础技能（Traj、Sit、Carry、Climb）之外，新增第五个基础交互技能 **HumanoidRun** —— 给定瞬时目标速度向量（朝向 + 速度大小），策略让根关节水平速度贴合目标。

**核心定位**：与 Traj 互补 —— Traj 控位置序列（积分级），Run 控瞬时速度（导数级）。任务观测从 Traj 的 30 维（10 个 waypoint × 3）压到 Run 的 3 维（heading_x, heading_y, target_speed），后期接入 stage1 Transformer 的 token 模板更紧凑。

---

## 2. 实验设计：W1 → W2 两阶段

| 阶段 | 数据 | speed 区间 | 目的 |
|---|---|---|---|
| **W1** | 仓库内已 retarget 的 12 段 walk（dataset_amass_loco/） | 0.5–1.5 m/s | 不依赖 AMASS 原始数据，先打通 pipeline |
| **W2** | W1 数据 + AMASS 中筛 18 段 jog/run（4 个 subset） | 0.5–3.5 m/s | 扩展到 sprint 速度，验证数据扩充的必要性 |

W2 的 18 段 jog/run 来自：
- ACCAD/Female1Running_c3d (6 段：C2/C3/C4/C5/C11/C13)
- ACCAD/Male2Running_c3d (2 段：C11/C17)
- CMU/09 (5 段：09_01/05/06/09/11，著名 run 主体)
- KIT/3 (2 段：walking_run02/04)
- HumanEva (3 段：S1/S2/S3 各 Jog_1)

---

## 3. 训练结果

| 阶段 | epochs | wallclock | 最终 reward (last 50–100) |
|---|---|---|---|
| W1 | 3308 | 1h 52min | **154.16 ± 2.09** |
| W2 | 4887 | 2h 52min | **104.03 ± 2.55** |

W2 reward 比 W1 低，因为速度区间宽 2.3×，平均 tracking 难度更大。

训练曲线（关键指标）：
- W1：reward 0.42 → 154，AMP disc loss 20.3 → 0.38
- W2：reward -2.9 → 104，AMP disc loss 0.99（plateau）

---

## 4. 评估：W1 vs W2 交叉 2×2 矩阵

每个 ckpt 在两个 speed range 下评估，每个 cell ≈ 76–98 episodes，episode 长度 299 步。

| Cell | ckpt | 测试速度 (m/s) | no-fall rate | mean reward (full ep) | est tracking err |
|---|---|---|---|---|---|
| **A** | W1 | 0.5 – 1.5 | **95.9%** | **203 ± 21** | **0.19** |
| B | W1 | 0.5 – 3.5 | 69.7% | 95 ± 68 | 0.57 |
| C | W2 | 0.5 – 1.5 | 88.2% | 161 ± 26 | 0.31 |
| **D** | W2 | 0.5 – 3.5 | **93.4%** | **132 ± 45** | **0.41** |

**关键发现：**

1. **Domain match 主导**：每个 ckpt 在自己的训练分布上最强（A、D 在矩阵对角线上是各自速度区间下的最优）。
2. **W1 在 sprint 速度下崩溃**：摔倒率从 4.1% 飙升至 30.3%（A→B），印证 jog/run 数据对高速训练不可或缺。
3. **W2 保留低速能力**：在低速区间仍 88.2% 不摔，但 reward 降低（203→161），存在 capacity interference 现象。
4. **W2 高速稳定性 vs W1**：在 0.5–3.5 区间，W2 的 93.4% 远胜 W1 的 69.7%。

**毕设论文章节素材**：表格（4 行 × 4 数据列）+ 上述结论 + 下方录像。

---

## 5. 工程产出清单

### 任务侧（Task 1–5）

| 文件 | 行数 | 类型 |
|---|---|---|
| `tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py` | 630 | 新建 |
| `tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml` | 71 | 新建 |
| `tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run_w1eval.yaml` | 71 | 新建（eval） |
| `tokenhsi/data/dataset_run/dataset_run.yaml` | 121 | 新建 |
| `tokenhsi/utils/parse_task.py` | +1 行 | 修改（注册 HumanoidRun） |
| `tokenhsi/scripts/single_task/run_train.sh` / `run_test.sh` / `run_test_save_video.sh` | 9 / 12 / 41 | 新建 |

### 数据流水线（Task 9–12）

| 文件 | 类型 |
|---|---|
| `tokenhsi/data/dataset_cfg.yaml` | 修改：新增 `motions.run` 节，18 段验证后的 AMASS 候选 |
| `tokenhsi/data/dataset_amass_run/preprocess.py` | 新建（复制自 dataset_amass_loco/ 改一行） |
| `tokenhsi/data/dataset_amass_run/generate_motion.py` | 新建（同上无字段差） |
| `tokenhsi/data/dataset_amass_run/check_amass_files.py` | 新建（A/B 工作分工 §6.6） |
| `tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py` | 新建（A/B 工作分工 §6.6） |
| `tokenhsi/data/dataset_amass_run/dataset_amass_run.yaml` | 自动产出 |
| `tokenhsi/data/data_utils.py` | 修改：兼容 SMPL+H G 的 `mocap_framerate` 字段（vs Phase II 的 `mocap_frame_rate`） |

### 评估归档

| 路径 | 内容 |
|---|---|
| `output/single_task/ckpt_run_w1.pth` | W1 checkpoint (40 MB) |
| `output/single_task/ckpt_run_w2.pth` | W2 checkpoint (40 MB) |
| `output/single_task/ckpt_run_w1.README.md` | W1 元数据 |
| `output/single_task/ckpt_run_w2.README.md` | W2 元数据 |
| `output/single_task/ckpt_run_w1.eval.log` | W1 in-domain eval log |
| `output/single_task/ckpt_run_w2.eval.log` | W2 in-domain eval log |
| `output/single_task/comp_w1_at_w2speed.log` | Cell B: W1 ckpt @ W2 速度 |
| `output/single_task/comp_w2_at_w1speed.log` | Cell C: W2 ckpt @ W1 速度 |

### 文档

- `docs/superpowers/specs/2026-05-05-humanoid-run-design.md` (设计文档 v2, 409 行)
- `docs/superpowers/plans/2026-05-05-humanoid-run.md` (实施计划, 1221 行)
- `docs/midterm/HUMANOID_RUN_SUMMARY.md` (本文档)

---

## 6. 数据流水线说明

```
[AMASS 原始 .npz, SMPL+H G 格式]
       ↓ preprocess.py (读 dataset_cfg.yaml 中的 motions.run 列表)
[smpl_params/<seq>.npy] (中间产物)
       ↓ generate_motion.py (SMPL skeleton → phys_humanoid_v3, retarget)
[motions/<subset>/<seq>/phys_humanoid_v3/ref_motion.npy]
       ↓ gen_dataset_yaml.py (扫描 motions/ 自动生成)
[dataset_amass_run.yaml]
       ↓ AMP 训练入口
```

实测：18 段从原始 AMASS 跑到 retargeted 产物总共 < 2 分钟。

---

## 7. 已知 issue 与未做事项

### 已知 issue（不阻塞当前结果）

1. **`successThreshold` eval 语义**：当前用速度跟踪误差实现（spec §2.3），但 eval 阶段 reward 与 train 不一致是 RL 项目通病。
2. **W2 在 reward < 0 的 episode**：偶发对极端速度目标无法跟随，未影响整体结论。
3. **chumpy 0.70 numpy 兼容性**：手动 patch 了 `from numpy import bool, ...` 一行（已在 commit message 记录）。
4. **dataset_cfg.yaml**：yaml.dump 重写时丢失了原有的注释行，仅保留结构。

### 未做事项

1. **录像（视频）**：当前服务器没有 X11 显示，未生成 `.mp4` / `.gif` 演示视频。可在有显示的机器上跑 `run_test_save_video.sh`。
2. **stage1 多任务集成**：spec §10 列出的 5 项整合工作（修改 `multi_task/humanoid_traj_sit_carry_climb.py` 等）属于毕设第二阶段，未在本次实施。
3. **更细粒度的 speed 扫描**：当前对照实验是随机采样 (0.5, 1.5) 和 (0.5, 3.5) 两个区间。如需在固定 target_speed ∈ {0.5, 1.0, 2.0, 3.0, 3.5} 下扫描，需小改 humanoid_run.py 的 `_sample_run_targets` 加一个 deterministic 选项。

---

## 8. Commit 链

```
e8852c7 feat(eval): W1 vs W2 对照实验 (2x2 cross-evaluation)
0ff64b2 feat(w2): W2 训练完成 + 评估归档
c7964ce feat(w2): dataset_run + amp_humanoid_run 升级到 W2
656e3d2 feat(data+w1): 完成 Phase 2 用户侧 + W1 评估
15ebc57 fix(data): dataset_cfg.yaml run: 节适配实际 AMASS SMPL+H G 命名
9e480a0 feat(data): 新增 dataset_amass_run/ 流水线
2c39e2e feat(data): dataset_cfg.yaml 新增 run: 节
2e871ef fix(task): humanoid_run.py 中残留的 HumanoidTraj.StateInit 引用
a6211b6 feat(scripts): run_train/test/test_save_video.sh
048cca3 feat(data): dataset_run.yaml (W1)
e4d6c4f feat(parse_task): 注册 HumanoidRun
0cef63b feat(cfg): amp_humanoid_run.yaml (W1)
f43bc87 feat(task): HumanoidRun 基础技能任务类
7e3fac6 docs(plan): 实施计划
70a7da3 docs(spec): 设计文档 v2
83ab117 docs: 设计文档 v1
```

15 个 commit 完整记录从 brainstorming → spec → plan → 实施 → 实验 → 归档全流程。

---

## 9. 复现指南

### 复现 W1（不需要 AMASS）
```bash
conda activate tokenhsi
cd /home/yued/TokenHSI
bash tokenhsi/scripts/single_task/run_train.sh
# 等 ~2h 后停止；checkpoint 在 output/Humanoid_<timestamp>/nn/Humanoid.pth
```

### 复现 W2（需要 AMASS）
```bash
# 步骤 1：下载 AMASS SMPL+H G 子集 (ACCAD/CMU/KIT/HumanEva), SMPL_NEUTRAL.pkl
# 步骤 2：编辑 tokenhsi/data/dataset_cfg.yaml 改 amass_dir 路径
# 步骤 3：跑数据流水线
python tokenhsi/data/dataset_amass_run/check_amass_files.py
python tokenhsi/data/dataset_amass_run/preprocess.py
python tokenhsi/data/dataset_amass_run/generate_motion.py
python tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py

# 步骤 4：编辑 dataset_run.yaml 把 walk 和 run motions 合并 (已自动化, 见 commit c7964ce)
# 步骤 5：训练
bash tokenhsi/scripts/single_task/run_train.sh    # ~3h
```

---

*— 实验总结结束 —*
