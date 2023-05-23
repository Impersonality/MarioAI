from gym import Wrapper


class CustomReward(Wrapper):
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.curr_score = 0
        self.current_x = 40

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        if info["x_pos"] <= self.current_x:
            reward -= 3
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return self.env.reset()
