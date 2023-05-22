import gym


class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomWrapper, self).__init__(env)

        self.curr_score = 0
        self.fitness_current = 0
        self.counter = 0
        self.score = 0
        self.score_tracker = 0
        self.coins = 0
        self.coins_tracker = 0
        self.yoshi_coins = 0
        self.yoshi_coins_tracker = 0
        self.x_pos_previous = 0
        self.y_pos_previous = 0
        self.checkpoint = False
        self.power_ups_last = 0

    def reset(self):
        self.curr_score = 0
        self.fitness_current = 0
        self.counter = 0
        self.score = 0
        self.score_tracker = 0
        self.coins = 0
        self.coins_tracker = 0
        self.yoshi_coins = 0
        self.yoshi_coins_tracker = 0
        self.x_pos_previous = 0
        self.y_pos_previous = 0
        self.checkpoint = False
        self.power_ups_last = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        score = info['score']
        coins = info['coins']
        yoshi_coins = info['yoshiCoins']
        dead = info['dead']
        x_pos = info['x']
        y_pos = info['y']
        jump = info['jump']
        checkpoint_value = info['checkpoint']
        end_of_level = info['endOfLevel']
        power_ups = info['powerups']
        done = False

        # Add to fitness score if mario gains points on his score.
        if score > 0:
            if score > self.score_tracker:
                self.fitness_current = (score * 10)
                self.score_tracker = score

        # Add to fitness score if mario gets more coins.
        if coins > 0:
            if coins > self.coins_tracker:
                self.fitness_current += (coins - self.coins_tracker)
                self.coins_tracker = coins

        # Add to fitness score if marioe gets more yoshi coins.
        if yoshi_coins > 0:
            if yoshi_coins > self.yoshi_coins_tracker:
                self.fitness_current += (yoshi_coins - self.yoshi_coins_tracker) * 10
                self.yoshi_coins_tracker = yoshi_coins

        # As mario moves right, reward him slightly.
        if x_pos > self.x_pos_previous:
            if jump > 0:
                self.fitness_current += 10
            self.fitness_current += (x_pos / 100)
            self.x_pos_previous = x_pos
            self.counter = 0
        # If mario is standing still or going backwards, penalize him slightly.
        else:
            self.counter += 1
            self.fitness_current -= 1

            # Award mario slightly for going up higher in the y position (y pos is inverted).
        if y_pos < self.y_pos_previous:
            self.fitness_current += 10
            self.y_pos_previous = y_pos
        elif y_pos < self.y_pos_previous:
            self.y_pos_previous = y_pos

        # If mario loses a powerup, punish him 1000 points.
        if power_ups == 0:
            if self.power_ups_last == 1:
                self.fitness_current -= 500
                print("Lost Upgrade")
        # If powerups is 1, mario got a mushroom...reward him for keeping it.
        elif power_ups == 1:
            if self.power_ups_last == 0:
                self.fitness_current += 10
            if self.power_ups_last == 1:
                self.fitness_current += 0.025
            elif self.power_ups_last == 2:
                self.fitness_current -= 500
                print("Lost Upgrade")
        # If powerups is 2, mario got a cape feather...reward him for keeping it.
        elif self.power_ups == 2:
            self.fitness_current += 0.05

        self.power_ups_last = power_ups

        # If mario reaches the checkpoint (located at around xpos == 2425) then give him a huge bonus.
        if checkpoint_value == 1 and not self.checkpoint:
            self.fitness_current += 20000
            self.checkpoint = True

        # If mario reaches the end of the level, award him automatic winner.
        if end_of_level == 1:
            self.fitness_current += 1000000
            done = True

        # If mario is standing still or going backwards for 1000 frames, end his try.
        if self.counter >= 500:
            self.fitness_current -= 175
            done = True

            # If mario dies, dead becomes 0, so when it is 0, penalize him and move on.
        if dead == 0:
            self.fitness_current -= 100
            done = True

        return obs, reward / 10. + self.fitness_current * 0.001, done, info
