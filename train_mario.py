import os
import gym
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def main():
    # 环境名称
    env_name = 'SuperMarioWorld-Snes'

    # 创建一个环境实例
    env = retro.make(env_name, state='Start')
    env = Monitor(env)

    # 使用stable_baselines3的DummyVecEnv包装环境
    # 这将使我们能够更轻松地使用stable_baselines3库
    env = DummyVecEnv([lambda: env])
    
    
    # 创建一个评估回调，这将定期评估模型并将结果写入 TensorBoard
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                             name_prefix='rl_model')
    # 创建一个新的PPO模型
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_mario_tensorboard/")

    # 在环境中训练模型
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # 保存模型
    model.save(f'{env_name}_ppo_model')

    # 关闭环境
    env.close()


if __name__ == '__main__':
    main()
