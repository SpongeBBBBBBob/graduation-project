# HumanoidRun：第五个基础交互技能 · 设计文档

- **作者**：毕设作者（与 AI 协助 brainstorm）
- **日期**：2026-05-05
- **状态**：设计已与项目作者确认，待写实现计划
- **关联工作**：TokenHSI 单任务训练框架（`tokenhsi/scripts/single_task/{traj,sit,carry,climb}_train.sh`）

---

## 1. 背景与动机

### 1.1 现状

TokenHSI 现有四个基础交互技能（basic interaction skills）：

| 技能 | 类 | 任务类型 | 是否需物体 |
|---|---|---|---|
| Traj | `HumanoidTraj` | 沿 2D 轨迹行走 | 否 |
| Sit | `HumanoidSit` | 坐到椅子上 | 是（StraightChair） |
| Carry | `HumanoidCarry` | 搬运箱子 | 是（Box） |
| Climb | `HumanoidClimb` | 攀爬到平台 | 是（Box/Cabinet/Table） |

毕设方向：在此之上**新增第五个基础技能**，并以 single-task 训练为第一步。

### 1.2 为什么选 Run/Sprint

候选方案对比（详见 brainstorming 过程）：

| 候选 | 是否需新物体 | 是否需新数据 | 工程量 |
|---|---|---|---|
| **Run / Sprint** | 否 | W1 完全复用现有数据；W2 需从 AMASS 筛 jog/run | ★☆☆☆☆ |
| GetUp（跌倒起身） | 否 | 是 | ★★☆☆☆ |
| Push（推箱子） | 复用 Box | 是 | ★★★☆☆ |
| LieDown（躺下） | 是（bed/sofa 新增） | 是 | ★★★☆☆ |
| Stairs | 是（楼梯地形） | 已有 | ★★★★☆，且与现有 adapt 任务冲突 |

**选定 Run/Sprint**，理由：
1. 不引入任何新资产，最低工程风险；
2. 与现有 4 技能正交（Traj 控位置序列，Run 控瞬时速度向量，文献里是两个独立任务）；
3. 任务观测维度小（3 维），后期接入 stage1 Transformer 时 token 设计干净；
4. **W1 阶段完全不需要 AMASS 原始数据**，可立刻开干；W2 数据准备和 W1 训练并行，零阻塞。

### 1.3 与 Traj 的明确区分（防答辩问"这跟 Traj 改两个数有什么区别"）

- **Traj**：跟踪未来 5s 内 10 个位置 waypoint（30 维任务观测），属于**积分级**控制。
- **Run**：跟踪当前瞬时目标速度向量（3 维任务观测），属于**导数级**控制；速度可从慢走（0.5 m/s）覆盖到冲刺（3.5 m/s），步态学习是核心。

任务观测形态、奖励函数、参考动作分布三者**全部**与 Traj 不同，不是简单的超参替换。

---

## 2. 任务定义

### 2.1 命名

- Task 类名：`HumanoidRun`
- 命令行：`--task HumanoidRun`
- 目录代号：`run`（小写）

### 2.2 Episode 结构

- **长度**：300 步 ≈ 10 s（30 Hz，与 Traj 一致）
- **早期终止**：`terminationHeight=0.15`（摔倒即终止，与所有现有任务一致）
- **不启用 IET**：Run 是持续性跟踪任务，没有"完成即停"的概念；这与 Traj 一致，与 Sit/Climb/Carry 不同。

### 2.3 成功条件（仅评估时使用）

- `success = (episode 内最后 60 步的平均 root velocity tracking error < 0.3 m/s)`
- 在训练 yaml 的 `eval:` 节下覆盖；训练阶段不读这个值。

---

## 3. 任务观测（Task Observation）

3 维，全部在 humanoid **局部坐标系（root 旋转后）**下：

| 维度 | 含义 | 取值范围 |
|---|---|---|
| `tar_heading_x` | 目标朝向单位向量 x | [-1, 1] |
| `tar_heading_y` | 目标朝向单位向量 y | [-1, 1]（约束 `x²+y²=1`） |
| `tar_speed` | 目标速度大小 (m/s) | `[speedMin, speedMax]` |

> 与 Traj 的 30 维任务观测（10 waypoints × 3）相比，3 维的紧凑表示在后期接入 stage1 Transformer 时，token 模板更简洁，跨任务对齐压力更小。

### 3.1 目标的时间演化（target schedule）

参考 Traj 的 `sharpTurnProb` 机制：

- Episode 起始：随机采样 `(tar_heading, tar_speed)`
  - `tar_heading`：方位角 ∈ [0, 2π) 均匀采样后转单位向量
  - `tar_speed`：在 `[speedMin, speedMax]` 均匀采样
- 每步以低概率 `targetChangeProb ≈ 0.005` 重新采样目标，制造方向/速度突变
- 这样既保证大部分时间稳态训练（policy 学到的是"长时间稳定跟踪"），又保证策略能应对目标切换

---

## 4. 奖励函数

```
v_target = tar_speed * [tar_heading_x, tar_heading_y]   # 2D 目标速度向量
v_actual = root_lin_vel.xy                              # 2D 实际水平速度
err = || v_actual - v_target ||₂                        # L2 误差

r_task  = exp( -2.0 * err )                             # 速度跟踪奖励，∈ (0, 1]
r_power = -0.0005 * action_power                        # 功率惩罚（与现有任务一致）

total_reward = r_task + r_power
```

**为什么用误差向量模长而不是拆 heading + speed 两项？**
- 单项 reward 形式简单，调试时易于诊断；
- 拆开会引入两个权重超参，且两项之间会争夺梯度；
- 作为 W1 baseline 优先用最简形式，若 W2 阶段发现高速跟不上再迭代为分项形式。

---

## 5. 终止条件与早期终止

| 条件 | 触发时机 | 处理 |
|---|---|---|
| 摔倒（`root_height < 0.15`） | 任意步 | early termination，episode 结束 |
| episode 长度到达 300 步 | 第 300 步 | 自然结束 |
| 不启用 IET（最大交互步） | 不适用 | 持续跟踪任务，无"完成即停"概念 |

---

## 6. 数据策略（W1 → W2 分阶段）

### 6.1 W1：用现有 walk 数据先打通 pipeline

- **数据集**：直接复用 `dataset_amass_loco`（18 段 walk，**仓库内已预处理好**，无需任何 AMASS 原始数据）
- **配置**：`speedMin=0.5, speedMax=1.5`（与 Traj 完全一致）
- **能力上限**：speed-controlled walking
- **验收标准**：tracking error 收敛至 < 0.3 m/s，肉眼看不会摔
- **风险评估**：≈ 0；用的是已经验证过 Traj 任务能跑通的数据

> ⚠️ 重要事实：仓库的 `tokenhsi/data/dataset_amass_loco/motions/` 已经包含**预处理后的 ref_motion.npy**，是从原始 AMASS 经 SMPL → phys_humanoid_v3 retarget 得到的最终产物。W1 阶段完全不需要碰 AMASS 原始数据。

### 6.2 W2：补 jog/run 数据，扩展速度上限

W2 阶段需要把 AMASS 中**未被现有四个技能用到的 jog/run 片段**补进数据集，因此**必须**走完整的 AMASS → SMPL → 骨架 retarget 流水线（详见 §6.4）。

#### 6.2.1 候选 AMASS 片段清单（带置信度）

> ⚠️ 文件名以 AMASS 官方 SMPL+H Phase II 命名为准。下面的清单中：
> - 🟢 **高置信**：项目其它技能已经用过同 subset（说明本地路径/命名一致），或 ACCAD/HumanEva 这种命名严格规范的子集
> - 🟡 **中置信**：subset 大概率有此类动作，但具体文件编号需在下载后核对
> - 🔴 **待验证**：纯凭印象，必须由"清单核对脚本"在 `<amass_dir>` 中实际扫描确认

```yaml
run:
  # ── ACCAD：命名严格规范，置信度高
  - "ACCAD+__+Female1Running_c3d+__+C3_-_Run_stageii.npz"            # 🟢
  - "ACCAD+__+Female1Running_c3d+__+C5_-_Walk_to_run_stageii.npz"    # 🟢
  - "ACCAD+__+Female1Running_c3d+__+C2_-_Run_to_stand_t2_stageii.npz"# 🟡（编号需确认）
  - "ACCAD+__+Male2Running_c3d+__+C3_-_run_stageii.npz"              # 🟢
  - "ACCAD+__+Male2Running_c3d+__+C9_-_run_to_walk_stageii.npz"      # 🟡

  # ── CMU：subset 已被 climb 任务用过 (路径肯定能找到)，但具体编号需确认
  - "CMU+__+09+__+09_05_stageii.npz"  # 🟡 subject 09 = run/jog 著名
  - "CMU+__+09+__+09_06_stageii.npz"  # 🟡
  - "CMU+__+09+__+09_09_stageii.npz"  # 🟡
  - "CMU+__+02+__+02_03_stageii.npz"  # 🔴 编号是猜的
  - "CMU+__+35+__+35_17_stageii.npz"  # 🔴 是否真为 jog 待确认

  # ── BMLrub：jogging 文件编号纯靠猜
  - "BMLrub+__+rub002+__+0027_jogging1_stageii.npz"  # 🔴
  - "BMLrub+__+rub075+__+0027_jogging1_stageii.npz"  # 🔴

  # ── HumanEva（备选）：有 Run/Jog，命名规范
  # - "HumanEva+__+S1+__+Jog_1_stageii.npz"  # 🟡（如果后续数据不够再补）
```

**应对策略**：
1. 由 §6.6 的 **B0 步**先在本地 AMASS 中扫描这份清单，把存在的留下（典型情况下能保留 6–10 段）
2. 6 段 jog/run 数据对 AMP 判别器训练是充足的下限
3. 如果保留不足 6 段，再扫 `<amass_dir>` 中含 `run|jog|sprint` 关键字的文件名做候选扩增

#### 6.2.2 训练配置变化

- `speedMax` 从 1.5 提到 3.5
- 训练策略：从零重训（速度分布变了）；可选 finetune 加速
- `dataset_run.yaml` 混合 walk（来自现成的 `dataset_amass_loco/motions/`）+ jog/run（来自新建的 `dataset_amass_run/motions/`），jog/run 略提采样权重（`weight: 1.5`）补偿数量

### 6.3 dataset_cfg.yaml 改动

在现有 `motions:` 下新增 `run:` 列表，列出 W2 候选 AMASS 文件名（项目 AMASS 文件白名单机制）。

### 6.4 AMASS → 训练数据 流水线（项目机制说明）

每个 `dataset_amass_<X>/` 子目录都是这条流水线的一份独立实例，**只有 `cfg["motions"][<key>]` 一行不同**：

```
┌──────────────────────────────────────────────────────────────┐
│ [AMASS 原始 .npz]  位于 <amass_dir>/<subset>/<subject>/...  │ ← 用户负责下载
│        ▼                                                    │
│  preprocess.py                                              │
│    读 dataset_cfg.yaml 中 motions.<key>                     │
│    把 "X+__+Y+__+Z_stageii.npz" 解码成路径                  │
│    process_amass_seq() 抽 SMPL 参数                         │
│        ▼                                                    │
│ [smpl_params/<seq>.npy]    （中间产物）                     │
│        ▼                                                    │
│  generate_motion.py                                         │
│    加载 SMPL body model（需 SMPL 权重文件）                 │
│    SMPL 骨架 → phys_humanoid_v3.xml 骨架 retarget           │
│    修足部穿地、根高度偏移                                   │
│        ▼                                                    │
│ [motions/<subset>/<seq>/phys_humanoid_v3/ref_motion.npy]    │ ← AMP 训练直接用
│        ▼                                                    │
│  gen_dataset_yaml.py（本设计新增的小工具）                  │
│        ▼                                                    │
│ [dataset_amass_run.yaml]   （AMP 训练入口）                 │
└──────────────────────────────────────────────────────────────┘
```

### 6.5 最小化 AMASS 下载方案（用户从零下载）

**前置条件**：用户当前**本地无 AMASS 原始数据**，需到 [AMASS 官网](https://amass.is.tue.mpg.de/) 注册并下载 SMPL+H G **Phase II** 数据。AMASS 全量约数百 GB，**不需要全量**。

| 优先级 | AMASS subset | 估算大小 | 用途 |
|---|---|---|---|
| **必下** | ACCAD | ~ 1.5 GB | 含 Female/Male Running c3d，`C*_-_Run_*` 系列 |
| **建议下** | HumanEva | ~ 200 MB | 含 Jog/Run，命名规范，是补充候选 |
| **建议下** | KIT | ~ 几 GB | 大量 walk/jog/run 片段（已被现有 loco 任务用过部分） |
| **可选** | CMU | ~ 30 GB | 含丰富 run motions（subject 09/16/49），但**最大** |
| **可选** | BMLrub | ~ 几 GB | 含 jogging 片段，但文件名编码不直观 |
| **跳过** | 其它 | — | 与 Run/Sprint 任务相关性低 |

**最小可行集**：仅下载 **ACCAD**（1.5 GB），就足够拿到 5–8 段 Run 数据，满足 W2 启动门槛。其它都是锦上添花。

**SMPL body model**：还需到 [SMPL 官网](https://smpl.is.tue.mpg.de/) 注册并下载 SMPL_NEUTRAL.pkl（~ 40 MB），放到项目要求的 `body_models/` 目录中（具体路径见 `body_models/model_loader.py` 的实现）。

### 6.6 §6.2 的 A/B 工作分工表

> 本节明确"AI 能做什么 / 用户必须做什么"，避免后期实施时混乱。

#### A. AI 可独立完成（不需要 AMASS 数据）

| ID | 工作内容 | 产出 |
|---|---|---|
| A1 | 创建 `tokenhsi/data/dataset_amass_run/` 目录脚手架 | `preprocess.py` + `generate_motion.py`（基本是 `dataset_amass_loco/` 的复制 + 一行 `cfg["motions"]["loco"]` → `cfg["motions"]["run"]` 修改） |
| A2 | 在 `tokenhsi/data/dataset_cfg.yaml` 新增 `run:` 候选清单 | §6.2.1 的带置信度清单 |
| A3 | 写 `tokenhsi/data/dataset_amass_run/check_amass_files.py` | 给定 `<amass_dir>`，扫描清单文件，输出 `[存在 / 缺失]` 报告，并产出"实际可用清单" |
| A4 | 写 `tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py` | 扫描 retarget 后的 `motions/` 目录，自动生成 `dataset_amass_run.yaml` |
| A5 | 准备 W2 阶段的 `dataset_run.yaml` 模板 | 占位 + walk 引用 + run 引用，run 部分会在 A4 完成后回填 |

#### B. 用户必须本地完成（依赖 AMASS / SMPL / GPU）

| ID | 工作内容 | 备注 |
|---|---|---|
| B0 | 注册 AMASS / SMPL 账号 | 一次性，免费 |
| B1 | 下载 AMASS 必需 subset 到本地 `<amass_dir>` | 最小 1.5 GB（ACCAD）；越多越好 |
| B2 | 下载 SMPL_NEUTRAL.pkl，配置 `body_models/` | 一次性 |
| B3 | 修改 `dataset_cfg.yaml` 的 `amass_dir` 字段 | 把占位符替换为真实路径 |
| B4 | 跑 `check_amass_files.py`（A3 提供）| 输出真实可用候选 |
| B5 | 跑 `preprocess.py` | 产出 `smpl_params/*.npy`，每段 < 10 s |
| B6 | 跑 `generate_motion.py` | 产出 `motions/*/ref_motion.npy`，每段约 1–3 分钟（CPU+GPU 混合，单线程） |
| B7 | 跑 `gen_dataset_yaml.py`（A4 提供） | 产出最终 `dataset_amass_run.yaml` |

---

## 7. 文件改动清单（"五件套"模板 + 数据流水线）

### 7.0 任务侧（W1 即可使用，无 AMASS 依赖）

| # | 文件 | 类型 | 行数估算 | 主要内容 |
|---|---|---|---|---|
| 1 | `tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py` | 新建 | ~350 | `HumanoidRun(Humanoid)` 类，参考 `humanoid_traj.py` 删减 |
| 2 | `tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml` | 新建 | ~75 | 复制 `amp_humanoid_traj.yaml`，移除 traj 相关，新增 run 相关 |
| 3 | `tokenhsi/data/dataset_run/dataset_run.yaml` | 新建 | ~40 | W1 仅列 walk；W2 追加 jog/run |
| 4 | `tokenhsi/utils/parse_task.py` | 修改 | +1 | 注册 `HumanoidRun` import |
| 5 | `tokenhsi/scripts/single_task/run_train.sh` | 新建 | 9 | 镜像 `traj_train.sh` |
| 6 | `tokenhsi/scripts/single_task/run_test.sh` | 新建 | ~12 | 镜像 `traj_test.sh` |
| 7 | `tokenhsi/scripts/single_task/run_test_save_video.sh` | 新建 | ~14 | 镜像 `traj_test_save_video.sh` |

### 7.1 数据流水线侧（W2 依赖，可与 W1 实现并行）

| # | 文件 | 类型 | 行数估算 | 主要内容 |
|---|---|---|---|---|
| 8 | `tokenhsi/data/dataset_cfg.yaml` | 修改 | +20 | 在 `motions:` 下新增 `run:` 列表（§6.2.1） |
| 9 | `tokenhsi/data/dataset_amass_run/preprocess.py` | 新建 | ~45 | 复制 `dataset_amass_loco/preprocess.py`，将 `cfg["motions"]["loco"]` 改为 `cfg["motions"]["run"]` |
| 10 | `tokenhsi/data/dataset_amass_run/generate_motion.py` | 新建 | ~220 | 与 `dataset_amass_loco/generate_motion.py` 几乎一致（无字段差异） |
| 11 | `tokenhsi/data/dataset_amass_run/check_amass_files.py` | 新建 | ~50 | A3 工具：扫描 `<amass_dir>` 验证候选清单文件存在性，输出报告 |
| 12 | `tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py` | 新建 | ~60 | A4 工具：扫描 `motions/` 目录自动产出 `dataset_amass_run.yaml` |
| 13 | `tokenhsi/data/dataset_amass_run/dataset_amass_run.yaml` | **由 #12 生成** | ~30 | 列出本地实际产出的 jog/run motions |

### 7.1 `humanoid_run.py` 从 `humanoid_traj.py` 改的关键点

最大化复用、最小化新代码：

1. **删除**所有 `traj_generator`、`_traj_samples`、`_fail_dist` 相关字段与方法；
2. **改 task_obs**：从"采样 N 个 waypoint 的 local 坐标"改为"3 维 (heading_x, heading_y, speed) ——但这 3 维的 heading 已是 root-local frame 下的方向向量，不需要再做变换"；
3. **改 reward**：从位置/朝向跟踪改为速度向量跟踪（exp(-2 \* err) + power penalty）；
4. **新增 buffer**：`self._tar_heading [num_envs, 2]`、`self._tar_speed [num_envs]`、`self._tar_change_prob`；
5. **新增方法**：`_sample_targets(env_ids)` 采样目标，`_compute_run_reward()` 计算奖励；
6. **删除方法**：`_update_traj`、`_compute_traj`、`_draw_task` 中的 trajectory 可视化（保留并替换为绘制目标速度箭头）。

### 7.2 `amp_humanoid_run.yaml` 关键差异（vs `amp_humanoid_traj.yaml`）

| 字段 | traj | run |
|---|---|---|
| `numTrajSamples` | 10 | **删除** |
| `trajSampleTimestep` | 0.5 | **删除** |
| `sharpTurnProb` | 0.02 | **删除** |
| `sharpTurnAngle` | 1.57 | **删除** |
| `accelMax` | 2.0 | **删除** |
| `targetChangeProb` | — | **新增** 0.005 |
| `speedMin` | 0.5 | 0.5 (W1) / 0.5 (W2) |
| `speedMax` | 1.5 | 1.5 (W1) / 3.5 (W2) |
| `skill` | `loco_walkonly` | `loco_walkonly` (W1) / `loco_run` (W2) |

`skill` 字段会影响 `humanoid.py` 中 motion library 的初始化，需要在 W2 阶段同步看 `humanoid.py` 的 `_skill` 处理逻辑确认是否需要新增 skill 类型。

---

## 8. 训练资源与时长估计

| 阶段 | num_envs | 预计 epoch | 单卡时长（A100 / RTX 4090） |
|---|---|---|---|
| W1 | 1024（与 traj 一致，稳） | 200–300 | 4–6 h |
| W2 | 1024 / 4096 | 400–500 | 8–10 h（1024）/ 4–5 h（4096） |

`headless` 模式下显存占用约 8–14 GB，主流单卡可承受。

---

## 9. 验证 / 测试方案

### 9.1 训练阶段
- **TensorBoard 监控**：`mean_reward`、`r_task`、`r_power`、`mean tracking error`
- **early debug**：训练前 10 个 iteration 看 loss 是否下降，policy 是否输出非饱和动作
- **AMP disc loss**：保持在 0.5–0.7 之间（典型健康范围）

### 9.2 评估阶段
- **eval 脚本**：`run_test.sh`（单速度），扫描多个 `tar_speed ∈ {0.5, 1.0, 2.0, 3.0, 3.5}` 各 N=64 个 episode，统计成功率与平均 tracking error
- **录像**：`run_test_save_video.sh` 录 5–10 段不同 (heading, speed) 组合，肉眼定性验证步态合理性

### 9.3 与 Traj 的对照实验（论文表格用）
- 在相同硬件、相同 epoch 数下，分别训 Traj 与 Run，记录收敛速度、最终 tracking error、step efficiency
- 跨任务测试：把 Run 的策略放到 Traj 任务（位置跟踪）上 zero-shot 测试，预期失败 → 印证两者是不同任务

---

## 10. Stage1 多任务集成（暂不实施，但预留接口）

后期把 Run 加入 stage1 多任务策略时需要做的事（**本设计仅记录、暂不实施**）：

1. 修改 `tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py` → 新建 `humanoid_traj_sit_carry_climb_run.py` 或扩展现类
2. 在 `tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml` 中新增 run 任务的 token 配置
3. 在 `learning/transformer/amp_network_builder_transformer.py` 中检查 task obs 维度切片
4. 数据混合：在 `dataset_loco_sit_carry_climb.yaml` 中加入 run 数据
5. stage1_train.sh 增加 run 任务 flag

> 这部分作为"接下来的工作"放到毕设第二阶段，写另一个独立的 spec。

---

## 11. 风险与开放问题

| 风险 | 概率 | 缓解 |
|---|---|---|
| W1 阶段策略收敛但 reward 长期低 | 中 | 检查 `r_task` 值域：err ≈ 0.3 时 r_task ≈ exp(-0.6) ≈ 0.55，是合理值；如果训练后期 mean reward < 0.4 说明跟踪误差仍 > 0.5 m/s，需要调 reward scale 系数 -2.0 |
| W2 阶段高速段（>2.5 m/s）跟踪不上 | 中 | 检查 jog/run 数据量是否足够（< 5 段会被 walk 数据淹没），必要时扩 jog/run 采样权重至 2.0–3.0 |
| AMASS 候选文件名与实际文件不匹配（§6.2.1 中 🔴 部分） | **中高** | 由 A3 的 `check_amass_files.py` 在下载后自动核对，缺失即剔除；若可用 < 6 段则用 §6.2.1 末尾的"关键字扫描"兜底 |
| 用户尚未下载 AMASS（W2 阻塞） | 当前确实存在 | W1 完全不依赖 AMASS，可立即开始；W2 数据准备与 W1 训练并行进行（user 下载 + 我提供脚本） |
| SMPL body model 未配置导致 generate_motion.py 失败 | 中 | 在 A3 的 check 脚本中同时校验 SMPL_NEUTRAL.pkl 路径有效性 |
| `humanoid.py` 中 `skill` 字段处理不兼容新值 | 低 | W1 阶段沿用 `loco_walkonly`，W2 阶段评估是否需要扩 skill 类型 |
| 后期接入 stage1 时 task obs 维度对齐问题 | 低 | 3 维任务观测有充足 padding 空间，token 模板设计时即可对齐 |

---

## 12. 交付物

- **代码**：§7 的 13 项改动（任务侧 7 项 + 数据流水线 6 项，含自动生成的 `dataset_amass_run.yaml`）
- **训练 checkpoint**：W1 一份，W2 一份
- **评估视频**：3–5 段 single-task 演示
- **TensorBoard 日志**：W1、W2 完整训练曲线
- **本文档**：作为论文实验章节"第五技能：Run/Sprint"的设计参考

---

## 13. 实施顺序（待 writing-plans 阶段细化）

W1 与 W2 数据准备**可并行**进行：W1 实现/训练**完全不依赖 AMASS**，而 W2 数据准备的最大瓶颈是用户下载与 retarget 计算，正好可以在 W1 训练的几小时空档中完成。

```
时间轴 →

  AI ──┬─ T1: W1 实现（任务侧 §7.0，文件 #1–#7）─────► T2: 启动 W1 训练 ─► T3: W1 评估
       │                                                                           │
       └─ T4: W2 数据脚手架（§7.1 文件 #8–#12）─┐                                   │
                                                ▼                                   │
  用户 ─────► U1: 下载 AMASS ACCAD + SMPL ─► U2: 跑 preprocess + generate ─► U3: 验证产物
                                                                            │
                                                          T5: AI 跑 gen_dataset_yaml.py
                                                                            │
                                                                            ▼
                                                          T6: W2 训练 ─► T7: W2 评估
                                                                            │
                                                                            ▼
                                                            T8: 对照实验 + 归档 docs/midterm/
```

具体 todo 列表与每步验收准则，将在下一阶段（writing-plans skill）输出。

---

*— 文档结束 —*
