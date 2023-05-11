import gym
import retro
from stable_baselines3 import PPO


def main():
    env_name = 'SuperMarioWorld-Snes'
    model_path = f'{env_name}_ppo_model'

    # 加载训练好的模型
    model = PPO.load(model_path)

    # 创建一个环境实例
    env = retro.make(env_name, state='Start')

    num_episodes = 10
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 使用训练好的模型来选择动作
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            # 累积奖励
            total_reward += reward

            # 渲染游戏画面，这样你就可以看到AI是如何玩游戏的
            env.render()

        print(f'Episode {episode + 1}: Total Reward = {total_reward}')

    # 关闭环境
    env.close()


if __name__ == '__main__':
    main()
