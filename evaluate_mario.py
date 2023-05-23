from stable_baselines3 import PPO
import gym_super_mario_bros


def main():
    model_path = 'mario_ppo_model'

    # 加载训练好的模型
    model = PPO.load(model_path)

    env = gym_super_mario_bros.make('SuperMarioBros-v2')

    num_episodes = 10
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 使用训练好的模型来选择动作
            action, _ = model.predict(obs.copy())
            obs, reward, done, info = env.step(action.item())

            # 累积奖励
            total_reward += reward

            # 渲染游戏画面，这样你就可以看到AI是如何玩游戏的
            env.render()

        print(f'Episode {episode + 1}: Total Reward = {total_reward}')

    # 关闭环境
    env.close()


if __name__ == '__main__':
    main()
