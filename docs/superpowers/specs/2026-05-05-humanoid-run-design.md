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
| **Run / Sprint** | 否 | 仅 W2 阶段需补 jog/run（AMASS 已有） | ★☆☆☆☆ |
| GetUp（跌倒起身） | 否 | 是 | ★★☆☆☆ |
| Push（推箱子） | 复用 Box | 是 | ★★★☆☆ |
| LieDown（躺下） | 是（bed/sofa 新增） | 是 | ★★★☆☆ |
| Stairs | 是（楼梯地形） | 已有 | ★★★★☆，且与现有 adapt 任务冲突 |

**选定 Run/Sprint**，理由：
1. 不引入任何新资产，最低工程风险；
2. 与现有 4 技能正交（Traj 控位置序列，Run 控瞬时速度向量，文献里是两个独立任务）；
3. 任务观测维度小（3 维），后期接入 stage1 Transformer 时 token 设计干净；
4. AMASS 已有充足 jog/run 数据，分阶段验证空间大。

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

- **数据集**：直接复用 `dataset_amass_loco`（18 段 walk，已预处理）
- **配置**：`speedMin=0.5, speedMax=1.5`（与 Traj 完全一致）
- **能力上限**：speed-controlled walking
- **验收标准**：tracking error 收敛至 < 0.3 m/s，肉眼看不会摔
- **风险评估**：≈ 0；用的是已经验证过 Traj 任务能跑通的数据

### 6.2 W2：补 jog/run 数据，扩展速度上限

**新筛 AMASS 候选片段**（最终以本地存在性为准）：

```yaml
run:
  # ACCAD running
  - "ACCAD+__+Female1Running_c3d+__+C2_-_Run_to_stand_t2_stageii.npz"
  - "ACCAD+__+Female1Running_c3d+__+C3_-_Run_stageii.npz"
  - "ACCAD+__+Female1Running_c3d+__+C5_-_Walk_to_run_stageii.npz"
  - "ACCAD+__+Male2Running_c3d+__+C3_-_run_stageii.npz"
  - "ACCAD+__+Male2Running_c3d+__+C9_-_run_to_walk_stageii.npz"
  # CMU jog/run
  - "CMU+__+02+__+02_03_stageii.npz"
  - "CMU+__+09+__+09_05_stageii.npz"
  - "CMU+__+09+__+09_06_stageii.npz"
  - "CMU+__+09+__+09_09_stageii.npz"
  - "CMU+__+35+__+35_17_stageii.npz"
  # BMLrub jogging
  - "BMLrub+__+rub002+__+0027_jogging1_stageii.npz"
  - "BMLrub+__+rub075+__+0027_jogging1_stageii.npz"
```

- **配置**：`speedMax` 从 1.5 提到 3.5
- **训练策略**：从零重训（速度分布变了）；可选 finetune 加速
- **dataset_run.yaml** 混合 walk + jog/run，jog/run 略提采样权重（`weight: 1.5`）补偿数量

### 6.3 dataset_cfg.yaml 改动

在现有 `motions:` 下新增 `run:` 列表，列出 W2 候选 AMASS 文件名（项目 AMASS 文件白名单机制）。

---

## 7. 文件改动清单（"五件套"模板）

| # | 文件 | 类型 | 行数估算 | 主要内容 |
|---|---|---|---|---|
| 1 | `tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py` | 新建 | ~350 | `HumanoidRun(Humanoid)` 类，参考 `humanoid_traj.py` 删减 |
| 2 | `tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml` | 新建 | ~75 | 复制 `amp_humanoid_traj.yaml`，移除 traj 相关，新增 run 相关 |
| 3 | `tokenhsi/data/dataset_run/dataset_run.yaml` | 新建 | ~40 | W1 仅列 walk；W2 追加 jog/run |
| 4 | `tokenhsi/data/dataset_cfg.yaml` | 修改 | +15 | 在 `motions:` 下新增 `run:` 列表 |
| 5 | `tokenhsi/utils/parse_task.py` | 修改 | +1 | 注册 `HumanoidRun` import |
| 6 | `tokenhsi/scripts/single_task/run_train.sh` | 新建 | 9 | 镜像 `traj_train.sh` |
| 7 | `tokenhsi/scripts/single_task/run_test.sh` | 新建 | ~12 | 镜像 `traj_test.sh` |
| 8 | `tokenhsi/scripts/single_task/run_test_save_video.sh` | 新建 | ~14 | 镜像 `traj_test_save_video.sh` |

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
| AMASS run 数据本地不全 | 低 | 用户确认本地有完整 AMASS；若个别文件缺失，从候选列表中删除即可，不阻塞 |
| `humanoid.py` 中 `skill` 字段处理不兼容新值 | 低 | W1 阶段沿用 `loco_walkonly`，W2 阶段评估是否需要扩 skill 类型 |
| 后期接入 stage1 时 task obs 维度对齐问题 | 低 | 3 维任务观测有充足 padding 空间，token 模板设计时即可对齐 |

---

## 12. 交付物

- **代码**：上述 8 个文件改动
- **训练 checkpoint**：W1 一份，W2 一份
- **评估视频**：3–5 段 single-task 演示
- **TensorBoard 日志**：W1、W2 完整训练曲线
- **本文档**：作为论文实验章节"第五技能：Run/Sprint"的设计参考

---

## 13. 实施顺序（待 writing-plans 阶段细化）

1. **预备**：改 `dataset_cfg.yaml` 新增 `run:` 节（先准备好以便随时 W2）
2. **W1 实现**（按文件清单 #1–#8 顺序）
3. **W1 训练**（≈ 4–6 h）
4. **W1 评估 + 录像 + 调试**
5. **W2 数据准备**（筛 AMASS 文件，更新 `dataset_run.yaml`）
6. **W2 训练**（≈ 8–10 h）
7. **W2 评估 + 录像 + 对照实验**
8. **整理实验材料归档至 `docs/midterm/`**

具体 todo 列表与每步验收准则，将在下一阶段（writing-plans skill）输出。

---

*— 文档结束 —*
