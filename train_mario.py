import gym_super_mario_bros
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from mario_wrapper import CustomReward
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def make_env(seed=0):
    def _init():
        env = gym_super_mario_bros.make('SuperMarioBros-v2')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = CustomReward(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


def linear_schedule(initial_value, final_value):
    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def main():
    num_envs = 6  # 根据你的CPU核心数量调整

    # model参数
    total_timesteps = 100000
    learning_rate_schedule = linear_schedule(2.5e-4, 2.5e-6)
    # learning_rate_schedule = 1e-4

    clip_range_schedule = linear_schedule(0.15, 0.025)
    # clip_range_schedule = 0.2

    # 使用SubprocVecEnv来创建并行环境
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # 创建一个评估回调，这将定期评估模型并将结果写入 TensorBoard
    checkpoint_callback = CheckpointCallback(save_freq=31250, save_path='./logs/', name_prefix='rl_model')

    # 创建一个新的PPO模型
    model = PPO('CnnPolicy', env, device='cuda', verbose=1, batch_size=64, n_steps=512, gamma=0.9, n_epochs=10,
                gae_lambda=1, ent_coef=0.01,
                tensorboard_log='./ppo_mario_tensorboard/',
                learning_rate=learning_rate_schedule,
                clip_range=clip_range_schedule)

    # # 加载最近保存的模型检查点
    # model = PPO.load('logs_1/rl_model_9937500_steps.zip', env)

    # 在环境中训练模型
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # 保存模型
    model.save('mario_ppo_model')

    # 关闭环境
    env.close()


if __name__ == '__main__':
    main()
