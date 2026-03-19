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
TokenHSI 主入口模块

本模块负责启动强化学习训练流程，支持 AMP（Adversarial Motion Priors）和
基于 Transformer 的策略。包含环境创建、算法注册、训练循环等核心逻辑。
"""

import os

# 工具模块：配置、参数解析、任务解析
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

# rl_games 框架：算法、环境、运行器
from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch

# AMP 相关：智能体、玩家、模型、网络构建器
from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder

# Transformer 变体网络构建器：多任务、组合、自适应、长时程
from learning.transformer import amp_network_builder_transformer
from learning.transformer import amp_network_builder_transformer_comp
from learning.transformer import amp_network_builder_transformer_adapt
from learning.transformer import amp_network_builder_transformer_longterm

# Transformer 智能体与玩家
from learning.transformer import trans_agent
from learning.transformer import trans_players

# 全局变量：命令行参数、环境配置、训练配置（在 main 中初始化）
args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    """
    创建 RLGPU 仿真环境。

    支持 Horovod 多 GPU 分布式训练。根据配置解析仿真参数、解析任务，
    并返回可用的并行环境实例。

    Returns:
        env: 创建好的强化学习环境
    """
    # 检查是否启用 Horovod 多 GPU 训练
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        # 为每个 rank 设置不同的随机种子，避免数据重复
        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        # 配置当前 rank 使用的 GPU 设备
        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    # 解析仿真参数并创建任务与环境
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))

    # 可选：帧堆叠（用于 Atari 等需要历史帧的任务）
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    """
    算法观察器，用于记录和输出训练过程中的成功率等统计信息。

    继承自 rl_games 的 AlgoObserver，在训练各阶段回调时统计连续成功次数，
    并写入 TensorBoard 以便可视化。
    什么是 TensorBoard？TensorBoard 是一个用于可视化 TensorFlow 模型的工具。
    TensorBoard 可以用于可视化模型的结构、参数、损失函数、训练进度等。
    """

    def __init__(self, use_successes=True):
        """use_successes: 是否使用 episode 成功标志，否则使用连续成功次数"""
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        """算法初始化完成后调用，保存算法引用和统计工具"""
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        """处理 episode 结束时的 info，更新连续成功次数统计"""
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        """清除统计后调用"""
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        """打印统计后调用，将连续成功次数写入 TensorBoard"""
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    """
    RLGPU 向量化环境封装。

    实现 rl_games 的 IVecEnv 接口，封装底层仿真环境。支持局部观测（obs）
    与全局状态（states）两种模式，用于 PPO 等算法的训练。
    """

    def __init__(self, config_name, num_actors, **kwargs):
        """根据配置名创建环境，并初始化 full_state（obs + 可选的 states）"""
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        """执行动作，返回下一观测、奖励、终止标志和 info"""
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        """重置环境，env_ids 为 None 时重置全部，否则只重置指定 id 的环境"""
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        """返回环境中的智能体数量"""
        return self.env.get_number_of_agents()

    def get_env_info(self):
        """返回环境信息字典，供算法构建网络时使用"""
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


# 向 rl_games 注册 RLGPU 向量化环境类型
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})


def build_alg_runner(algo_observer):
    """
    构建算法运行器，注册 AMP 与 Transformer 相关的算法、玩家和网络。

    Returns:
        runner: 配置好的 Runner 实例
    """
    runner = Runner(algo_observer)

    # 注册 AMP 算法及其组件
    runner.algo_factory.register_builder('amp', lambda **kwargs : amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

    # 注册 Transformer 算法及其多种网络架构
    runner.algo_factory.register_builder('trans', lambda **kwargs : trans_agent.TransAgent(**kwargs))
    runner.player_factory.register_builder('trans', lambda **kwargs : trans_players.TransPlayerContinuous(**kwargs))
    runner.model_builder.network_factory.register_builder('amp_transformer_multi_task', lambda **kwargs : amp_network_builder_transformer.AMPTransformerMultiTaskBuilder())
    runner.model_builder.network_factory.register_builder('amp_transformer_multi_task_comp', lambda **kwargs : amp_network_builder_transformer_comp.AMPTransformerMultiTaskCompBuilder())
    runner.model_builder.network_factory.register_builder('amp_transformer_multi_task_adapt', lambda **kwargs : amp_network_builder_transformer_adapt.AMPTransformerMultiTaskAdaptBuilder())
    runner.model_builder.network_factory.register_builder('amp_transformer_multi_task_longterm', lambda **kwargs : amp_network_builder_transformer_longterm.AMPTransformerMultiTaskLongTermTaskCompletionBuilder())

    return runner

def main():
    """
    主入口：解析配置、初始化环境、构建运行器并启动训练。
    """
    global args
    global cfg
    global cfg_train

    # 设置 NumPy 输出格式
    set_np_formatting()
    args = get_args() # 获取命令行参数
    cfg, cfg_train, logdir = load_cfg(args) # 加载配置

    # 设置随机种子以保证可复现性
    cfg_train['params']['seed'] = cfg_train['params']['config']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    # 命令行覆盖配置
    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod # 是否使用horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length # 设置horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size # 设置minibatch_size

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    # 创建默认的权重与统计输出目录
    cfg_train['params']['config']['train_dir'] = args.output_path

    vargs = vars(args) # 将命令行参数转换为字典

    # 创建算法观察器并构建运行器
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return


if __name__ == '__main__':
    main()
