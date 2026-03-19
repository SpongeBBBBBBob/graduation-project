# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
配置文件工具模块 (config.py)

本模块负责 TokenHSI 项目的配置管理，主要功能包括：
- 加载和合并 YAML 配置文件（训练配置 + 环境配置）
- 解析 Isaac Gym 仿真参数
- 解析命令行参数
- 设置随机种子以保证实验可复现性
"""

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

# 仿真时间步长：每帧 1/60 秒，对应 60Hz 的物理仿真频率
SIM_TIMESTEP = 1.0 / 60.0

def set_np_formatting():
    """
    设置 NumPy 数组的打印格式。

    用于调试时更好地查看大型数组内容：
    - edgeitems: 每维首尾各显示 30 个元素
    - linewidth: 每行最大 4000 字符
    - precision: 浮点数保留 2 位小数
    - threshold: 超过 10000 个元素时使用省略号
    """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    """
    任务名称校验失败时抛出异常。

    当传入的 task 参数不在支持的任务列表中时调用，
    提示用户可用的任务类型。
    """
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    """
    设置随机种子，保证实验可复现性。

    Args:
        seed: 随机种子。若为 -1 且 torch_deterministic=True 则使用 42；
              若为 -1 则随机生成 0-10000 之间的整数
        torch_deterministic: 是否启用 PyTorch 完全确定性模式（会降低性能）

    Returns:
        int: 实际使用的种子值

    注意：torch_deterministic=True 时会设置 CUBLAS 工作空间配置，
    参考 https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    """
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    # 设置 Python、NumPy、PyTorch 的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # 完全确定性模式：牺牲性能换取可复现性
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        # 默认模式：启用 cudnn benchmark 以提升训练速度
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):
    """
    加载并合并配置文件，将命令行参数覆盖到 YAML 配置中。

    从两个 YAML 文件加载配置：
    - cfg_train: 训练相关配置（PPO 超参数、网络结构等）
    - cfg_env: 环境相关配置（仿真参数、任务设置等）

    Args:
        args: 命令行解析得到的参数对象

    Returns:
        tuple: (cfg, cfg_train, logdir)
            - cfg: 环境配置（已与命令行参数合并）
            - cfg_train: 训练配置（已与命令行参数合并）
            - logdir: 日志输出目录
    """
    # 加载训练配置 YAML
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f: # 加载训练配置文件
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    # 加载环境配置 YAML
    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f: # 加载环境配置文件
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # 命令行参数覆盖：环境数量
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    # 命令行参数覆盖：回合长度
    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    # 任务名称与无头模式
    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # 物理域随机化（Domain Randomization）：用于提升策略的泛化能力
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir

    # 确定性模式：保证训练过程可复现
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    # 实验名称：可由 --experiment 和 --metadata 组合生成
    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment

    # 将生成的实验名称写回配置
    cfg_train["params"]["config"]['name'] = exp_name

    # 从检查点恢复训练
    if args.resume > 0:
        cfg_train["params"]["load_checkpoint"] = True

    # 指定检查点路径（用于加载预训练权重或评估）
    if args.checkpoint != "Base":
        cfg_train["params"]["load_path"] = args.checkpoint
        cfg["env"]["checkpoint"] = args.checkpoint

    # 分层强化学习（HRL）：低层控制器（LLC）检查点路径，支持最多 7 个
    if args.llc_checkpoint != "":
        cfg_train["params"]["config"]["llc_checkpoint"] = args.llc_checkpoint

        if args.llc_checkpoint_2 != "":
            cfg_train["params"]["config"]["llc_checkpoint_2"] = args.llc_checkpoint_2
        
        if args.llc_checkpoint_3 != "":
            cfg_train["params"]["config"]["llc_checkpoint_3"] = args.llc_checkpoint_3
        
        if args.llc_checkpoint_4 != "":
            cfg_train["params"]["config"]["llc_checkpoint_4"] = args.llc_checkpoint_4
        
        if args.llc_checkpoint_5 != "":
            cfg_train["params"]["config"]["llc_checkpoint_5"] = args.llc_checkpoint_5
        
        if args.llc_checkpoint_6 != "":
            cfg_train["params"]["config"]["llc_checkpoint_6"] = args.llc_checkpoint_6
        
        if args.llc_checkpoint_7 != "":
            cfg_train["params"]["config"]["llc_checkpoint_7"] = args.llc_checkpoint_7

    # 分层强化学习：高层控制器（HRL）检查点路径
    if args.hrl_checkpoint != "":
        cfg_train["params"]["config"]["hrl_checkpoint"] = args.hrl_checkpoint

    # 评估模式：仅在 --test 且 --eval 同时开启时启用
    if args.eval and args.test:
        cfg_train["params"]["config"]["eval"] = True
        cfg_train["params"]["config"]["eval_task"] = args.eval_task
    else:
        cfg_train["params"]["config"]["eval"] = False
        cfg_train["params"]["config"]["eval_task"] = args.eval_task

    # 命令行覆盖：最大训练迭代轮数（epochs）
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    # 同步环境数量到训练配置（actor 数量 = 并行环境数）
    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]
    # 策略网络推理设备（如 cuda:0）
    cfg_train["params"]["config"]["device"] = args.rl_device

    # 渲染设备与任务规划配置文件路径
    cfg["env"]["render_device"] = args.render_device
    cfg["env"]["task_plan"] = args.cfg_task_plan

    # 随机种子：命令行优先于配置文件
    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    # 将完整 args 对象存入 cfg，供后续使用
    cfg["args"] = args

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    """
    解析 Isaac Gym 仿真参数，构建 SimParams 对象。

    根据物理引擎类型（PHYSX 或 Flex）设置相应的仿真参数，
    支持从 cfg["sim"] 中读取 YAML 配置进行覆盖。

    Args:
        args: 命令行参数
        cfg: 环境配置字典
        cfg_train: 训练配置字典（本函数中未使用，保留接口一致性）

    Returns:
        gymapi.SimParams: Isaac Gym 仿真参数对象
    """
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.slices

    # Flex 物理引擎配置（软体仿真，较少使用）
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    # PhysX 物理引擎配置（刚体仿真，默认使用）
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024  # 800 万对接触

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # 若 YAML 中存在 sim 配置节，则解析并覆盖上述默认值
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 命令行覆盖：PhysX 求解器线程数
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False):
    """
    解析命令行参数，定义项目自定义参数并合并 Isaac Gym 内置参数。

    Args:
        benchmark: 若为 True，则额外添加基准测试相关参数（--num_proc, --random_actions 等）

    Returns:
        argparse.Namespace: 解析后的参数对象，包含 train/test/play 等运行模式标志
    """
    # 自定义命令行参数列表（按功能分组）
    custom_parameters = [
        # --- 运行模式 ---
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        # --- 任务与配置 ---
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_env", "type": str, "default": "Base", "help": "Environment configuration file (.yaml)"},
        {"name": "--cfg_train", "type": str, "default": "Base", "help": "Training configuration file (.yaml)"},
        {"name": "--cfg_task_plan", "type": str, "default": "", "help": "Task planning configuration file path"},

        {"name": "--motion_file", "type": str,
            "default": "", "help": "Specify reference motion file"},
        # --- 训练超参数 ---
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--horizon_length", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--output_path", "type": str, "default": "output/", "help": "Specify output directory"},
        # --- 分层强化学习（HRL）检查点 ---
        {"name": "--llc_checkpoint", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--llc_checkpoint_2", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--llc_checkpoint_3", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--llc_checkpoint_4", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--llc_checkpoint_5", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--llc_checkpoint_6", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--llc_checkpoint_7", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--hrl_checkpoint", "type": str, "default": "",
            "help": "Path to the saved weights for the high-level controller of an HRL agent."},

        {"name": "--render_device", "type": str, "default": "",
            "help": "Choose GPU device for rendering height map"},
        # --- 评估与录制 ---
        {"name": "--eval", "action": "store_true", "default": False,
            "help": "Enable evaluation mode when running with --test"},
        {"name": "--eval_task", "type": str, "default": "",
            "help": "Task name for evaluation"},
        {"name": "--record", "action": "store_true", "default": False,
            "help": "Record evaluation results or videos"},
        {"name": "--save_frames", "action": "store_true", "default": False,
            "help": "Save rendered frames to disk, then exit (for X11 forwarding)"},
        {"name": "--max_frames", "type": int, "default": 300,
            "help": "Max frames to save when --save_frames is set"},
        
        ]

    # 基准测试模式：追加性能测试相关参数
    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # 调用 Isaac Gym 工具解析命令行（合并内置参数与自定义参数）
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # 设备别名：与 Isaac Gym 示例保持一致
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    # 运行模式：test/play 为推理模式，否则为训练模式
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
