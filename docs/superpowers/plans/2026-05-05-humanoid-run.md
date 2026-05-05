# HumanoidRun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 TokenHSI 现有四个基础交互技能之外，新增第五个基础技能 `HumanoidRun`（瞬时速度向量跟踪），并完成 W1（walk-only 启动）和 W2（jog/run 扩展）两阶段单任务训练。

**Architecture:** 复用 `HumanoidTraj` 的 AMP + 任务奖励训练框架，将任务观测从"未来 10 个 waypoints (30 维)"改为"目标朝向 + 目标速度 (3 维)"，奖励改为速度向量 L2 误差的指数。W1 阶段直接复用仓库内已预处理的 walk motions；W2 阶段新增 AMASS 数据流水线 (`dataset_amass_run/`) 处理 jog/run 片段。

**Tech Stack:** Isaac Gym, rl_games (AMP), PyTorch, lpanlib (skeleton retargeting), SMPL body model；下游 SMPL_NEUTRAL.pkl + AMASS Phase II SMPL+H 数据。

**Reference Spec:** [`docs/superpowers/specs/2026-05-05-humanoid-run-design.md`](../specs/2026-05-05-humanoid-run-design.md)

---

## Overview

Plan 分三个 Phase：

- **Phase 1 (W1)**：纯任务侧实现，**完全不依赖 AMASS**，跑通 walk-only baseline。Task 1–8。
- **Phase 2 (W2 数据)**：搭建 AMASS 数据流水线，用户下载并跑预处理。**可与 Task 7（W1 训练）并行**。Task 9–17。
- **Phase 3 (W2 训练 + 实验)**：扩展速度上限到 sprint，跑 W2 训练，做对照实验。Task 18–24。

### File Structure

```
tokenhsi/
├── env/tasks/basic_interaction_skills/
│   └── humanoid_run.py                              [新建, ~350 行]
├── data/
│   ├── cfg/basic_interaction_skills/
│   │   └── amp_humanoid_run.yaml                    [新建, ~75 行]
│   ├── dataset_run/
│   │   └── dataset_run.yaml                         [新建, ~40 行]
│   ├── dataset_amass_run/                           [新建目录]
│   │   ├── preprocess.py                            [新建]
│   │   ├── generate_motion.py                       [新建]
│   │   ├── check_amass_files.py                     [新建]
│   │   ├── gen_dataset_yaml.py                      [新建]
│   │   ├── dataset_amass_run.yaml                   [由脚本生成]
│   │   ├── smpl_params/                             [流水线产物]
│   │   └── motions/                                 [流水线产物]
│   └── dataset_cfg.yaml                             [修改, +20 行]
├── utils/parse_task.py                              [修改, +1 行]
└── scripts/single_task/
    ├── run_train.sh                                 [新建]
    ├── run_test.sh                                  [新建]
    └── run_test_save_video.sh                       [新建]
```

### Branch Strategy

建议在 `main` 上做一个 feature 分支 `feature/humanoid-run`，每个 Phase 完成后合到 main。但当前仓库 main 已有未提交改动，可保留在 main 上做（用 commit 隔离即可）。

---

## Phase 0: 预飞检查

### Task 0: 验证项目状态 + 创建工作分支

**Files:**
- Modify: 无（仅 git 操作）

- [ ] **Step 1：确认 git 状态干净**

```bash
cd /home/yued/TokenHSI && /usr/bin/git status
```

期望：当前 main 分支，看到本计划文档的改动以及之前已存在的 modified 文件（不阻塞）。

- [ ] **Step 2：检查 Isaac Gym 环境可用**

```bash
cd /home/yued/TokenHSI && python -c "from isaacgym import gymapi; print('isaacgym OK')"
```

期望：`isaacgym OK`，无 import 错误。

- [ ] **Step 3：试跑现有 traj 任务确认环境**（约 30 秒，仅 5 epoch）

跳过 —— 假设你已经能跑 `traj_train.sh`。如果本步失败，先修环境再开始本计划。

---

## Phase 1: W1 任务侧实现（不依赖 AMASS）

### Task 1: 创建 `humanoid_run.py`

**Files:**
- Create: `tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py`
- Reference: `tokenhsi/env/tasks/basic_interaction_skills/humanoid_traj.py`

**Approach:** 这是本计划的核心代码改动。从 `humanoid_traj.py` 复制 → 删 traj generator → 改 task_obs / reward → 加 target sampling 逻辑。

- [ ] **Step 1：复制 `humanoid_traj.py` 作为起点**

```bash
cd /home/yued/TokenHSI && cp tokenhsi/env/tasks/basic_interaction_skills/humanoid_traj.py \
                            tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py
```

- [ ] **Step 2：批量重命名类名**

在 `humanoid_run.py` 中将 `HumanoidTraj` 全部改为 `HumanoidRun`。

- [ ] **Step 3：删除 trajectory 相关字段（在 `__init__`）**

从 `__init__` 中移除：
```python
self._num_traj_samples = cfg["env"]["numTrajSamples"]
self._traj_sample_timestep = cfg["env"]["trajSampleTimestep"]
self._sharp_turn_prob = cfg["env"]["sharpTurnProb"]
self._sharp_turn_angle = cfg["env"]["sharpTurnAngle"]
self._fail_dist = 4.0
```

新增：
```python
self._tar_change_prob = cfg["env"]["targetChangeProb"]   # 例 0.005
```

保留：
```python
self._speed_min = cfg["env"]["speedMin"]
self._speed_max = cfg["env"]["speedMax"]
self._enable_task_obs = cfg["env"]["enableTaskObs"]
self._power_reward = cfg["env"]["power_reward"]
self._power_coefficient = cfg["env"]["power_coefficient"]
```

- [ ] **Step 4：把 `_build_traj_generator` 替换为 `_build_run_targets`**

在 `__init__` 末尾把 `self._build_traj_generator()` 改为 `self._build_run_targets()`，并新增方法：

```python
def _build_run_targets(self):
    self._tar_heading = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
    self._tar_speed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
    env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._sample_run_targets(env_ids)
    return

def _sample_run_targets(self, env_ids):
    n = env_ids.shape[0]
    theta = torch.rand(n, device=self.device) * 2.0 * np.pi
    self._tar_heading[env_ids, 0] = torch.cos(theta)
    self._tar_heading[env_ids, 1] = torch.sin(theta)
    self._tar_speed[env_ids] = torch.rand(n, device=self.device) * (self._speed_max - self._speed_min) + self._speed_min
    return
```

- [ ] **Step 5：改 `get_task_obs_size`**

```python
def get_task_obs_size(self):
    obs_size = 0
    if (self._enable_task_obs):
        obs_size += 3   # heading_x, heading_y, target_speed
    return obs_size
```

- [ ] **Step 6：改 `_compute_task_obs`**

```python
def _compute_task_obs(self, env_ids=None):
    if (env_ids is None):
        root_states = self._humanoid_root_states
        tar_heading = self._tar_heading
        tar_speed = self._tar_speed
    else:
        root_states = self._humanoid_root_states[env_ids]
        tar_heading = self._tar_heading[env_ids]
        tar_speed = self._tar_speed[env_ids]

    # transform target heading from world to humanoid local frame
    obs = compute_run_observations(root_states, tar_heading, tar_speed)
    return obs
```

并在文件末尾新增：

```python
@torch.jit.script
def compute_run_observations(root_states, tar_heading, tar_speed):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    tar_heading_3d = torch.cat([tar_heading, torch.zeros_like(tar_heading[:, :1])], dim=-1)
    local_tar_heading = quat_rotate(heading_rot, tar_heading_3d)
    obs = torch.cat([local_tar_heading[:, :2], tar_speed.unsqueeze(-1)], dim=-1)
    return obs
```

- [ ] **Step 7：删除 `_fetch_traj_samples` 与 `_compute_traj_obs`**

整个方法删除（即 humanoid_traj.py 第 254–280 行附近的逻辑）。

- [ ] **Step 8：改 `_compute_reward` 中调用的 helper 函数**

把 `_compute_traj_reward` 替换为 `_compute_run_reward`，新增：

```python
def _compute_run_reward(self):
    root_lin_vel = self._humanoid_root_states[:, 7:10]
    v_target_x = self._tar_speed * self._tar_heading[:, 0]
    v_target_y = self._tar_speed * self._tar_heading[:, 1]
    err_x = root_lin_vel[:, 0] - v_target_x
    err_y = root_lin_vel[:, 1] - v_target_y
    err = torch.sqrt(err_x * err_x + err_y * err_y + 1e-8)
    r_task = torch.exp(-2.0 * err)
    if self._power_reward:
        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
        r_power = -self._power_coefficient * power
    else:
        r_power = torch.zeros_like(r_task)
    self.rew_buf[:] = r_task + r_power
    return
```

- [ ] **Step 9：改 `post_physics_step` 中的 target 切换逻辑**

在 `post_physics_step` 末尾追加：

```python
# random target change
mask = torch.rand(self.num_envs, device=self.device) < self._tar_change_prob
change_ids = torch.nonzero(mask, as_tuple=False).flatten()
if change_ids.shape[0] > 0:
    self._sample_run_targets(change_ids)
```

- [ ] **Step 10：改 `_reset_task`**

```python
def _reset_task(self, env_ids):
    self._sample_run_targets(env_ids)
    return
```

- [ ] **Step 11：删除 marker（trajectory 可视化）相关方法**

删除 `_load_marker_asset`、`_build_marker`、`_build_marker_state_tensors`、`_update_marker`，以及 `_create_envs` / `_build_env` 中调用它们的分支。本期不实现速度可视化箭头（论文图录像时用别的方法即可）。

- [ ] **Step 12：删除评估相关字段中 `successThreshold` 的位置距离语义**

`_is_eval` 分支保留，但 `successThreshold` 改为速度跟踪 success 阈值（语义为 m/s）。具体的 success 计算逻辑可暂留 `traj` 版本，W1 训练通过后再优化（暂不阻塞）。

- [ ] **Step 13：核验文件能否被 import**

```bash
cd /home/yued/TokenHSI && python -c "
import sys; sys.path.insert(0, 'tokenhsi')
from env.tasks.basic_interaction_skills.humanoid_run import HumanoidRun
print('HumanoidRun OK')
"
```

期望：`HumanoidRun OK`，无 syntax/import 错误。如果失败，回去修。

- [ ] **Step 14：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/env/tasks/basic_interaction_skills/humanoid_run.py
/usr/bin/git commit -F - <<'EOF'
feat(task): 新增 HumanoidRun 基础技能任务类

从 HumanoidTraj 改造而来。改动：
- 任务观测从 30 维 traj waypoints 改为 3 维 (heading_x, heading_y, target_speed)
- 奖励改为速度向量 L2 误差 + power penalty: r = exp(-2*err) - 0.0005*power
- 删除 traj_generator / marker 相关代码
- 新增 _sample_run_targets / _build_run_targets / _compute_run_reward
- post_physics_step 中以 targetChangeProb=0.005 概率随机切换目标
EOF
```

---

### Task 2: 创建 `amp_humanoid_run.yaml`

**Files:**
- Create: `tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml`
- Reference: `tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml`

- [ ] **Step 1：复制 traj 配置**

```bash
cd /home/yued/TokenHSI && cp tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
                            tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml
```

- [ ] **Step 2：删除 traj 专属字段**

从 `env:` 节下删除：
- `numTrajSamples: 10`
- `trajSampleTimestep: 0.5`
- `accelMax: 2.0`
- `sharpTurnProb: 0.02`
- `sharpTurnAngle: 1.57`

- [ ] **Step 3：新增 run 专属字段**

在 `env:` 节中新增（放在 `speedMax` 后）：
```yaml
  targetChangeProb: 0.005   # 每步切换目标的概率
```

- [ ] **Step 4：W1 阶段保持 `speedMin: 0.5, speedMax: 1.5`**

确认这两个值与 traj 一致。W2 阶段会改 `speedMax` 到 3.5。

- [ ] **Step 5：保持 `skill: "loco_walkonly"`**

W1 阶段沿用，与 dataset_amass_loco 中的 motion 索引一致。

- [ ] **Step 6：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml
/usr/bin/git commit -m "feat(cfg): 新增 amp_humanoid_run.yaml (W1 配置, speedMax=1.5)"
```

---

### Task 3: 注册 HumanoidRun

**Files:**
- Modify: `tokenhsi/utils/parse_task.py:32-37`

- [ ] **Step 1：在 basic_interaction_skills 的 import 块中加一行**

```python
from env.tasks.basic_interaction_skills.humanoid_run import HumanoidRun
```

放在现有 `humanoid_climb` import 之后。

- [ ] **Step 2：核验注册成功**

```bash
cd /home/yued/TokenHSI && python -c "
import sys; sys.path.insert(0, 'tokenhsi')
import utils.parse_task
print('parse_task imports OK')
"
```

- [ ] **Step 3：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/utils/parse_task.py
/usr/bin/git commit -m "feat(parse_task): 注册 HumanoidRun task class"
```

---

### Task 4: 创建 W1 阶段的 `dataset_run.yaml`

**Files:**
- Create: `tokenhsi/data/dataset_run/dataset_run.yaml`

W1 阶段：直接引用 `dataset_amass_loco/motions/` 下的 walk 文件。

- [ ] **Step 1：创建目录**

```bash
cd /home/yued/TokenHSI && mkdir -p tokenhsi/data/dataset_run
```

- [ ] **Step 2：写 W1 yaml**

复制 `tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml` 的内容，把所有 `motions/ACCAD/...` 路径改为相对当前 yaml 位置的 `../dataset_amass_loco/motions/ACCAD/...`：

```bash
cd /home/yued/TokenHSI && python <<'PYEOF'
import yaml
with open('tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml') as f:
    cfg = yaml.safe_load(f)
for skill_name in cfg['motions']:
    for entry in cfg['motions'][skill_name]:
        entry['file'] = '../dataset_amass_loco/' + entry['file']
with open('tokenhsi/data/dataset_run/dataset_run.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
print('Done.')
PYEOF
```

- [ ] **Step 3：核验路径解析正确**

```bash
cd /home/yued/TokenHSI && python <<'PYEOF'
import yaml, os
with open('tokenhsi/data/dataset_run/dataset_run.yaml') as f:
    cfg = yaml.safe_load(f)
base = 'tokenhsi/data/dataset_run/'
all_ok = True
for skill_name in cfg['motions']:
    for entry in cfg['motions'][skill_name]:
        path = os.path.normpath(os.path.join(base, entry['file']))
        if not os.path.isfile(path):
            print('MISSING:', path); all_ok = False
print('All OK' if all_ok else 'FAILED')
PYEOF
```

期望：`All OK`，每段 walk 的 ref_motion.npy 都存在。

- [ ] **Step 4：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_run/dataset_run.yaml
/usr/bin/git commit -m "feat(data): 新增 dataset_run.yaml (W1: 复用 walk motions)"
```

---

### Task 5: 创建训练/测试 shell 脚本

**Files:**
- Create: `tokenhsi/scripts/single_task/run_train.sh`
- Create: `tokenhsi/scripts/single_task/run_test.sh`
- Create: `tokenhsi/scripts/single_task/run_test_save_video.sh`

- [ ] **Step 1：写 `run_train.sh`**

```bash
#!/bin/bash

python ./tokenhsi/run.py --task HumanoidRun \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml \
    --motion_file tokenhsi/data/dataset_run/dataset_run.yaml \
    --num_envs 1024 \
    --headless
```

- [ ] **Step 2：写 `run_test.sh`**

```bash
#!/bin/bash

python ./tokenhsi/run.py --task HumanoidRun \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml \
    --motion_file tokenhsi/data/dataset_run/dataset_run.yaml \
    --checkpoint output/single_task/ckpt_run.pth \
    --test \
    --num_envs 16
```

- [ ] **Step 3：写 `run_test_save_video.sh`**

模仿 `traj_test_save_video.sh`，把 `HumanoidTraj` 改为 `HumanoidRun`、cfg/yaml 改为 run 版本。

- [ ] **Step 4：赋可执行权限**

```bash
cd /home/yued/TokenHSI && chmod +x tokenhsi/scripts/single_task/run_*.sh
```

- [ ] **Step 5：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/scripts/single_task/run_*.sh
/usr/bin/git commit -m "feat(scripts): 新增 run_train/test/test_save_video.sh"
```

---

### Task 6: W1 烟雾测试（5 epoch 快测）

**Goal:** 验证整条链路能跑通，不卡死、不崩溃。

- [ ] **Step 1：先用极小 num_envs 跑 5 个 iteration**

修改 `run_train.sh` 临时备份后，把 `num_envs` 改为 64，并在命令末尾加 `--max_iterations 5`（或在 cfg_train 中临时减小 max_epochs）。

```bash
cd /home/yued/TokenHSI && python ./tokenhsi/run.py --task HumanoidRun \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml \
    --motion_file tokenhsi/data/dataset_run/dataset_run.yaml \
    --num_envs 64 \
    --max_iterations 5 \
    --headless 2>&1 | tee /tmp/run_w1_smoke.log
```

期望：5 个 iteration 后程序自然退出，日志中能看到 `frame: ...`、`reward: ...` 行，且没有 NaN。

- [ ] **Step 2：检查日志**

```bash
grep -E "(reward|frame|loss|nan|NaN|Error|Traceback)" /tmp/run_w1_smoke.log | tail -30
```

期望：看到正数 reward 与正在下降的 loss；不应有 `NaN` 或 `Traceback`。

- [ ] **Step 3：失败排查（如适用）**

常见问题：
- `KeyError: 'targetChangeProb'` → 检查 yaml
- `RuntimeError: shape mismatch` → 检查 obs 维度（task_obs 应为 3）
- `task class not found` → 检查 parse_task.py
- 训练能跑但 reward 长期为 0 → 检查 `_compute_run_reward` 中速度索引（root_states[:, 7:10] 是否真的是 lin_vel）

- [ ] **Step 4：smoke 通过后不 commit（这是验证步骤）**

---

### Task 7: W1 完整训练（**长任务，4–6 小时**）

**Goal:** 训练 W1 baseline 至 tracking error < 0.3 m/s。

- [ ] **Step 1：启动后台训练**

```bash
cd /home/yued/TokenHSI && bash tokenhsi/scripts/single_task/run_train.sh 2>&1 | tee output/run_w1_train.log &
```

**注意**：请记录 `output/Humanoid_<timestamp>/` 目录名，下面会引用。

- [ ] **Step 2：5–10 分钟后初步检查**

```bash
tail -50 output/run_w1_train.log
```

期望：看到 `frame: 100000+`、`reward: 正数`、AMP disc loss 在 0.5–0.7 之间。

- [ ] **Step 3：启动 TensorBoard 监控**（可选，浏览器看曲线）

```bash
cd /home/yued/TokenHSI && tensorboard --logdir output/ --port 6006
```

- [ ] **Step 4：等到训练自然结束**（≈ 4–6 h）或当 mean_reward 稳定不再上升

收敛标志：
- `mean reward` 稳定 > 0.6
- `r_task` 稳定 > 0.5（对应 err ≈ 0.35 m/s 内）

- [ ] **Step 5：保存 W1 checkpoint 到稳定位置**

```bash
cd /home/yued/TokenHSI && mkdir -p output/single_task
cp output/Humanoid_<timestamp>/nn/Humanoid.pth output/single_task/ckpt_run_w1.pth
```

- [ ] **Step 6：归档训练日志**

```bash
cd /home/yued/TokenHSI && cp output/Humanoid_<timestamp>/random_seed.txt output/single_task/ckpt_run_w1.seed.txt
```

---

### Task 8: W1 评估 + 录像

**Goal:** 验证 W1 在多个 target_speed 下的 tracking error，录像目视确认步态。

- [ ] **Step 1：修改 `run_test.sh`，把 checkpoint 指向 W1**

```bash
sed -i 's|output/single_task/ckpt_run.pth|output/single_task/ckpt_run_w1.pth|' \
    tokenhsi/scripts/single_task/run_test.sh
```

- [ ] **Step 2：跑 eval（默认随机 speed）**

```bash
cd /home/yued/TokenHSI && bash tokenhsi/scripts/single_task/run_test.sh 2>&1 | tee /tmp/run_w1_eval.log
```

观察：是否出现摔倒、tracking error 大致水平。

- [ ] **Step 3：录像（最少 1 段，最多 3 段）**

```bash
cd /home/yued/TokenHSI && bash tokenhsi/scripts/single_task/run_test_save_video.sh
```

输出：`output/imgs/<timestamp>/video.mp4`。

- [ ] **Step 4：把 W1 视频 + log 归档**

```bash
cd /home/yued/TokenHSI && mkdir -p output/single_task/eval_w1
cp output/imgs/<latest>/video.mp4 output/single_task/eval_w1/run_w1.mp4
cp /tmp/run_w1_eval.log output/single_task/eval_w1/eval.log
```

- [ ] **Step 5：commit checkpoint+视频外的标记文件**（实际 .pth/.mp4 太大不入库，但写一行 README）

```bash
cd /home/yued/TokenHSI && cat > output/single_task/ckpt_run_w1.README.md <<'EOF'
# W1 Run Checkpoint
- Trained: <date>
- Source: tokenhsi/scripts/single_task/run_train.sh
- Data: dataset_run.yaml (W1: walk only, speedMax=1.5)
- Final mean reward: <value>
- Final mean tracking error: <value>
EOF

/usr/bin/git add output/single_task/ckpt_run_w1.README.md
/usr/bin/git commit -m "docs: W1 run checkpoint metadata"
```

---

## Phase 2: W2 数据流水线（与 Task 7 训练并行）

### Task 9: 在 `dataset_cfg.yaml` 新增 `run:` 候选清单

**Files:**
- Modify: `tokenhsi/data/dataset_cfg.yaml`

- [ ] **Step 1：在 `motions:` 下追加 `run:` 节**

参考 spec §6.2.1，在 `motions:` 下追加：

```yaml
  run:
    # ── ACCAD：高置信
    - "ACCAD+__+Female1Running_c3d+__+C3_-_Run_stageii.npz"
    - "ACCAD+__+Female1Running_c3d+__+C5_-_Walk_to_run_stageii.npz"
    - "ACCAD+__+Female1Running_c3d+__+C2_-_Run_to_stand_t2_stageii.npz"
    - "ACCAD+__+Male2Running_c3d+__+C3_-_run_stageii.npz"
    - "ACCAD+__+Male2Running_c3d+__+C9_-_run_to_walk_stageii.npz"
    # ── CMU：中置信，编号待 check_amass_files.py 验证
    - "CMU+__+09+__+09_05_stageii.npz"
    - "CMU+__+09+__+09_06_stageii.npz"
    - "CMU+__+09+__+09_09_stageii.npz"
    - "CMU+__+02+__+02_03_stageii.npz"
    - "CMU+__+35+__+35_17_stageii.npz"
    # ── BMLrub：低置信，凭印象
    - "BMLrub+__+rub002+__+0027_jogging1_stageii.npz"
    - "BMLrub+__+rub075+__+0027_jogging1_stageii.npz"
```

- [ ] **Step 2：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_cfg.yaml
/usr/bin/git commit -m "feat(data): 在 dataset_cfg.yaml 新增 run: 候选清单"
```

---

### Task 10: 创建 `dataset_amass_run/` 脚手架

**Files:**
- Create: `tokenhsi/data/dataset_amass_run/preprocess.py`
- Create: `tokenhsi/data/dataset_amass_run/generate_motion.py`

- [ ] **Step 1：创建目录**

```bash
cd /home/yued/TokenHSI && mkdir -p tokenhsi/data/dataset_amass_run
```

- [ ] **Step 2：复制并修改 `preprocess.py`**

```bash
cp tokenhsi/data/dataset_amass_loco/preprocess.py \
   tokenhsi/data/dataset_amass_run/preprocess.py
sed -i 's|cfg\["motions"\]\["loco"\]|cfg["motions"]["run"]|' \
   tokenhsi/data/dataset_amass_run/preprocess.py
```

- [ ] **Step 3：复制 `generate_motion.py`（无需改动）**

```bash
cp tokenhsi/data/dataset_amass_loco/generate_motion.py \
   tokenhsi/data/dataset_amass_run/generate_motion.py
```

- [ ] **Step 4：核验**

```bash
diff tokenhsi/data/dataset_amass_loco/preprocess.py \
     tokenhsi/data/dataset_amass_run/preprocess.py
```

期望：仅一行差异（`loco` → `run`）。

- [ ] **Step 5：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_amass_run/
/usr/bin/git commit -m "feat(data): 新增 dataset_amass_run/ 流水线脚手架（preprocess + generate_motion）"
```

---

### Task 11: 写 `check_amass_files.py`

**Files:**
- Create: `tokenhsi/data/dataset_amass_run/check_amass_files.py`

**Goal:** 给定 `<amass_dir>`，扫描 `dataset_cfg.yaml` 的 `run:` 候选清单，输出存在/缺失报告，并产出"实际可用清单"供后续 preprocess 用。

- [ ] **Step 1：写脚本内容**

完整内容：

```python
"""
Verify AMASS file existence for the `run` motion list.
Usage: python tokenhsi/data/dataset_amass_run/check_amass_files.py
"""
import os
import sys
import yaml

DATASET_CFG = os.path.join(os.path.dirname(__file__), "../dataset_cfg.yaml")
OUTPUT_AVAILABLE = os.path.join(os.path.dirname(__file__), "available_files.txt")
OUTPUT_MISSING = os.path.join(os.path.dirname(__file__), "missing_files.txt")


def main():
    with open(DATASET_CFG, "r") as f:
        cfg = yaml.safe_load(f)
    amass_dir = cfg["amass_dir"]
    if amass_dir.startswith("/YOUR_PATH"):
        print("[ERROR] dataset_cfg.yaml 中 amass_dir 仍为占位符，请先填真实路径")
        sys.exit(1)
    if not os.path.isdir(amass_dir):
        print(f"[ERROR] amass_dir 不存在: {amass_dir}")
        sys.exit(1)
    if "run" not in cfg["motions"]:
        print("[ERROR] dataset_cfg.yaml 中没有 motions.run 节")
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
        print("       建议运行关键字扫描脚本（待用户提供需求时再写）。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_amass_run/check_amass_files.py
/usr/bin/git commit -m "feat(data): 新增 check_amass_files.py，验证 AMASS 候选文件存在性"
```

---

### Task 12: 写 `gen_dataset_yaml.py`

**Files:**
- Create: `tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py`

**Goal:** 扫描 `motions/` 目录下所有 `phys_humanoid_v3/ref_motion.npy`，自动生成 `dataset_amass_run.yaml`。

- [ ] **Step 1：写脚本内容**

```python
"""
Auto-generate dataset_amass_run.yaml by scanning motions/ directory.
Usage: python tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py
"""
import os
import yaml
import glob

THIS_DIR = os.path.dirname(__file__)
MOTIONS_DIR = os.path.join(THIS_DIR, "motions")
OUTPUT_YAML = os.path.join(THIS_DIR, "dataset_amass_run.yaml")
SKILL_KEY = "loco_run"   # 与 amp_humanoid_run.yaml 中 skill 字段对齐


def main():
    if not os.path.isdir(MOTIONS_DIR):
        raise SystemExit(f"motions/ 目录不存在: {MOTIONS_DIR}，请先跑 generate_motion.py")

    pattern = os.path.join(MOTIONS_DIR, "*", "*", "phys_humanoid_v3", "ref_motion.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"未找到任何 ref_motion.npy 文件，请先跑 generate_motion.py")

    entries = []
    for f in files:
        rel = os.path.relpath(f, THIS_DIR)
        entries.append({"file": rel, "weight": 1.0})

    cfg = {"motions": {SKILL_KEY: entries}}
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {len(entries)} motions to {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py
/usr/bin/git commit -m "feat(data): 新增 gen_dataset_yaml.py，自动产出 dataset_amass_run.yaml"
```

---

### Task 13: **【用户操作】** 下载 AMASS ACCAD 与 SMPL_NEUTRAL.pkl

**Files:** 无（用户在浏览器/命令行操作）

- [ ] **Step 1：注册 AMASS 账号**

访问 https://amass.is.tue.mpg.de/ 注册，下载页面选择 **SMPL+H G** -> **Phase II (XXX)**。

- [ ] **Step 2：下载 ACCAD 子集 .tar.bz2 (~ 1.5 GB)**

解压到选定的 `<amass_dir>`，目录结构应为：
```
<amass_dir>/ACCAD/Female1Running_c3d/C3_-_Run_stageii.npz
<amass_dir>/ACCAD/Male2Running_c3d/...
```

- [ ] **Step 3：（可选）补 CMU 或 HumanEva 子集**

- [ ] **Step 4：注册 SMPL 账号**

访问 https://smpl.is.tue.mpg.de/ 注册，下载 SMPL_NEUTRAL.pkl。

- [ ] **Step 5：放置 SMPL 模型到正确位置**

根据 `body_models/model_loader.py:13` 的实现：

```python
body_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smpl")
```

需要把 SMPL_NEUTRAL.pkl 放到 `body_models/smpl/SMPL_NEUTRAL.pkl`：

```bash
cd /home/yued/TokenHSI && mkdir -p body_models/smpl
mv ~/Downloads/SMPL_NEUTRAL.pkl body_models/smpl/
ls -la body_models/smpl/
```

- [ ] **Step 6：在 `dataset_cfg.yaml` 中配置 `amass_dir`**

```bash
sed -i 's|/YOUR_PATH/datasets/AMASS|<your-actual-path>|' tokenhsi/data/dataset_cfg.yaml
```

把 `<your-actual-path>` 替换为你的实际 AMASS 目录绝对路径。

- [ ] **Step 7：核验**

```bash
ls $(grep amass_dir tokenhsi/data/dataset_cfg.yaml | awk '{print $2}' | tr -d '"')
```

期望：看到 `ACCAD/`、`CMU/` 等子目录。

---

### Task 14: **【用户操作】** 运行 `check_amass_files.py`

- [ ] **Step 1：扫描候选清单**

```bash
cd /home/yued/TokenHSI && python tokenhsi/data/dataset_amass_run/check_amass_files.py
```

期望：每条候选输出 `[OK]` 或 `[MISS]`。可用清单写入 `available_files.txt`。

- [ ] **Step 2：检查可用清单数量**

```bash
wc -l tokenhsi/data/dataset_amass_run/available_files.txt
```

- [ ] **Step 3：决策**

| 可用数 | 处理 |
|---|---|
| ≥ 8 | OK，可以直接进入 Task 15 |
| 6–7 | 边界，建议补下载 CMU 或 HumanEva |
| < 6 | 不够，必须补下载 |

- [ ] **Step 4：（如需）扩充候选清单**

如果 < 6 段，告知 AI 来扩充候选清单（再加入 KIT、HumanEva 等命名规范的候选），然后回到 Task 14 再扫一次。

---

### Task 15: **【用户操作】** 运行 `preprocess.py`

- [ ] **Step 1：修改 preprocess 以读 `available_files.txt`（推荐）**

为防止读 `dataset_cfg.yaml` 时把缺失文件也加进去，建议先临时把 `dataset_cfg.yaml` 的 `run:` 节替换为 `available_files.txt` 的内容。或：直接接受 preprocess.py 在缺失文件上跑出 ERROR（不会阻塞已存在文件的处理）。

简单做法：手动把 `dataset_cfg.yaml` 中的 `run:` 节裁剪为 `available_files.txt` 的内容。

- [ ] **Step 2：跑 preprocess**

```bash
cd /home/yued/TokenHSI && python tokenhsi/data/dataset_amass_run/preprocess.py
```

期望：输出 `Processed N sequences!`，`smpl_params/` 下生成 N 个 `.npy`。

- [ ] **Step 3：核验**

```bash
ls tokenhsi/data/dataset_amass_run/smpl_params/ | wc -l
```

期望：与 `available_files.txt` 行数一致。

---

### Task 16: **【用户操作】** 运行 `generate_motion.py`

**Note:** 这一步**最耗时**。每段 1–3 分钟，单线程，混合 CPU + GPU。10 段约 15–30 分钟。

- [ ] **Step 1：跑 retarget**

```bash
cd /home/yued/TokenHSI && python tokenhsi/data/dataset_amass_run/generate_motion.py
```

期望：进度条逐段处理，最终 `motions/<subset>/<seq>/phys_humanoid_v3/ref_motion.npy` 生成。

- [ ] **Step 2：核验**

```bash
find tokenhsi/data/dataset_amass_run/motions -name "ref_motion.npy" | wc -l
```

期望：与 smpl_params 下的 .npy 数量一致。

- [ ] **Step 3：（可选）抽一段看可视化 HTML**

打开 `tokenhsi/data/dataset_amass_run/motions/<subset>/<seq>/phys_humanoid_v3/ref_motion_render.html` 看 retarget 是否合理。

---

### Task 17: 运行 `gen_dataset_yaml.py` 产出最终 yaml

- [ ] **Step 1：跑生成脚本**

```bash
cd /home/yued/TokenHSI && python tokenhsi/data/dataset_amass_run/gen_dataset_yaml.py
```

期望：`Wrote N motions to .../dataset_amass_run.yaml`。

- [ ] **Step 2：核验**

```bash
cat tokenhsi/data/dataset_amass_run/dataset_amass_run.yaml | head -30
```

应看到 `motions: loco_run: [...]` 结构。

- [ ] **Step 3：commit（含产出的 yaml；motions/.npy 不入库）**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_amass_run/dataset_amass_run.yaml
echo "smpl_params/" > tokenhsi/data/dataset_amass_run/.gitignore
echo "motions/" >> tokenhsi/data/dataset_amass_run/.gitignore
echo "available_files.txt" >> tokenhsi/data/dataset_amass_run/.gitignore
echo "missing_files.txt" >> tokenhsi/data/dataset_amass_run/.gitignore
/usr/bin/git add tokenhsi/data/dataset_amass_run/.gitignore
/usr/bin/git commit -m "data: 产出 dataset_amass_run.yaml; 数据产物加入 gitignore"
```

---

## Phase 3: W2 训练 + 对照实验

### Task 18: 更新 `dataset_run.yaml` 为 W2（混合 walk + run）

- [ ] **Step 1：扩展 W1 yaml**

```bash
cd /home/yued/TokenHSI && python <<'PYEOF'
import yaml
with open('tokenhsi/data/dataset_run/dataset_run.yaml') as f:
    cfg = yaml.safe_load(f)
with open('tokenhsi/data/dataset_amass_run/dataset_amass_run.yaml') as f:
    run_cfg = yaml.safe_load(f)
# 把 loco_run 的 motion 加到 walkonly 同级，路径改为相对当前位置
for entry in run_cfg['motions']['loco_run']:
    entry['file'] = '../dataset_amass_run/' + entry['file']
    entry['weight'] = 1.5  # 略提高 jog/run 采样权重
cfg['motions']['loco_run'] = run_cfg['motions']['loco_run']
with open('tokenhsi/data/dataset_run/dataset_run.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
print('Done.')
PYEOF
```

- [ ] **Step 2：核验**

```bash
grep -c 'loco_walkonly\|loco_run' tokenhsi/data/dataset_run/dataset_run.yaml
```

期望：≥ 2（两个 skill 都存在）。

- [ ] **Step 3：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/dataset_run/dataset_run.yaml
/usr/bin/git commit -m "feat(data): dataset_run.yaml 升级到 W2（walk + jog/run 混合）"
```

---

### Task 19: 更新 `amp_humanoid_run.yaml` 为 W2 配置

- [ ] **Step 1：改 speedMax**

```bash
sed -i 's/speedMax: 1.5/speedMax: 3.5/' \
    tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml
```

- [ ] **Step 2：改 skill 字段**

```yaml
skill: ["loco_walkonly", "loco_run"]
skillInitProb: [0.3, 0.7]   # 多采 jog/run 起始
skillDiscProb: [0.3, 0.7]
```

如果 `humanoid_run.py` 没有处理多 skill 的逻辑，需要参考 `humanoid_sit.py` 的 skill 列表处理方式补一段（这是 W2 的隐藏工程量，建议在 smoke test 时一并核实）。

- [ ] **Step 3：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml
/usr/bin/git commit -m "feat(cfg): amp_humanoid_run.yaml 升级到 W2 (speedMax=3.5, skill=[walk,run])"
```

---

### Task 20: W2 烟雾测试

同 Task 6 的步骤，但检查重点不同：
- 高速段（target_speed=3.0+）的 reward 是否非零
- AMP disc loss 是否仍稳定

- [ ] **Step 1：5 epoch smoke test**

```bash
cd /home/yued/TokenHSI && python ./tokenhsi/run.py --task HumanoidRun \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_run.yaml \
    --motion_file tokenhsi/data/dataset_run/dataset_run.yaml \
    --num_envs 64 \
    --max_iterations 5 \
    --headless 2>&1 | tee /tmp/run_w2_smoke.log
```

- [ ] **Step 2：检查 multi-skill motion library 是否加载成功**

```bash
grep -E "(loco_walkonly|loco_run|skill|num_motion)" /tmp/run_w2_smoke.log | head -20
```

期望：看到两个 skill 各加载若干 motion 文件，无 KeyError。

---

### Task 21: W2 完整训练（**长任务，8–10 小时**）

同 Task 7 的流程，记录 `output/Humanoid_<timestamp>/` 路径：

- [ ] **Step 1：启动训练**
- [ ] **Step 2：监控 TensorBoard**
- [ ] **Step 3：等待收敛**
- [ ] **Step 4：保存 checkpoint 为 `output/single_task/ckpt_run_w2.pth`**

---

### Task 22: W2 评估 + 录像

- [ ] **Step 1：在多个 target_speed 下 eval**

跑 5 次 eval，每次 num_envs=16，target_speed 通过修改 yaml 中 `eval` 节的 speedMin/speedMax 控制（或临时改 eval 重写）：

```
target_speed ∈ {0.5, 1.0, 2.0, 3.0, 3.5}
```

记录每档下的：success_rate、mean_tracking_error、mean_episode_length。

- [ ] **Step 2：录像 3–5 段**

`run_test_save_video.sh` 跑多次，每次手动改 cfg 中 target speed。

- [ ] **Step 3：归档**

```bash
cd /home/yued/TokenHSI && mkdir -p output/single_task/eval_w2
cp output/imgs/*/video.mp4 output/single_task/eval_w2/
```

---

### Task 23: 对照实验（论文表格）

- [ ] **Step 1：W1 vs W2 vs Traj 在 Run 任务上对比**

直接用 `ckpt_run_w1.pth` / `ckpt_run_w2.pth` / `ckpt_traj.pth`（如有）跑同一组 eval，记录：
- success rate（按 §2.3 定义）
- mean tracking error
- average episode length

填入：

| Policy | Train Data | speed_max | success @ 1.5 | success @ 3.0 | err @ 1.5 | err @ 3.0 |
|---|---|---|---|---|---|---|
| Run-W1 | walk only | 1.5 | ? | ? | ? | ? |
| Run-W2 | walk + jog/run | 3.5 | ? | ? | ? | ? |
| Traj | walk only | 1.5 | ? (zero-shot) | ? | ? | ? |

- [ ] **Step 2：写实验小节**

输出：`docs/midterm/HUMANOID_RUN_RESULTS.md`，包含上表 + 视频链接 + TensorBoard 截图。

---

### Task 24: 归档实验材料

- [ ] **Step 1：把训练日志、视频、表格归档到 `docs/midterm/`**

```bash
cd /home/yued/TokenHSI && mkdir -p docs/midterm/run_experiment
cp output/single_task/eval_w*/video.mp4 docs/midterm/run_experiment/
cp output/single_task/ckpt_run_w*.README.md docs/midterm/run_experiment/
```

- [ ] **Step 2：写一份 `docs/midterm/HUMANOID_RUN_SUMMARY.md`**

简要总结：实验目标、最终结果、训练曲线、视频示例。可以参考 `STAGE1_EXPERIMENT_CHANGES.md` 的格式。

- [ ] **Step 3：commit**

```bash
cd /home/yued/TokenHSI && /usr/bin/git add docs/midterm/run_experiment/ docs/midterm/HUMANOID_RUN_SUMMARY.md
/usr/bin/git commit -m "docs(midterm): 归档 HumanoidRun 实验材料 (W1/W2 视频 + 总结)"
```

---

## Open Questions / TODOs

这些是计划中**目前留给执行阶段决策**的点：

1. `skill` 字段在 W2 阶段是 `["loco_walkonly", "loco_run"]` 还是改为别的命名？需看 `humanoid.py` 的 motion library 加载机制是否依赖特定 skill 名。
2. `successThreshold` 的语义在 Run 任务下应该是"速度跟踪误差" (m/s) 而不是"位置距离" (m)，需在 humanoid_run.py 的 eval 分支单独实现。
3. W2 训练是否要从 W1 checkpoint finetune？默认从零重训，但如果时间不够可以试 finetune。

---

## Verification Checklist (跑通即毕设可交付)

- [ ] W1 训练收敛，mean reward > 0.6
- [ ] W1 评估视频肉眼合理（不摔、能跟随目标速度）
- [ ] W2 数据流水线完整跑通，至少 6 段 jog/run 入库
- [ ] W2 训练收敛，target_speed=3.0 下 tracking error < 0.5 m/s
- [ ] W1/W2/Traj 对照实验表格完成
- [ ] `docs/midterm/HUMANOID_RUN_SUMMARY.md` 已写

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-humanoid-run.md`.

**两种执行方式：**

1. **Subagent-Driven（推荐用于 Phase 1 代码改动）**：每个 Task 派一个新 subagent 单独执行，完成后我 review 再放下一个。优点：subagent context 干净，错误隔离强。缺点：每次 spawn subagent 有 overhead。
2. **Inline 执行**（推荐用于 Phase 2/3 含长训练任务）：在当前 session 内逐 Task 执行，长训练任务用 backgrounded shell + AwaitShell。

由于本计划包含**长 RL 训练（数小时）**，建议混合模式：
- Phase 1（Task 1–6）：inline 执行，因为代码改动需要密切配合
- Task 7（W1 训练）：backgrounded
- Phase 2（Task 9–12 AI 部分）：inline，与 Task 7 并行
- Task 13–17（用户部分）：用户自己跑
- Task 18–24：inline + backgrounded 训练

---

*— 计划文档结束 —*
