import gym
import numpy as np
from gym import wrappers
from functools import reduce


class DiscreteCarEnvironment(gym.Wrapper):
    # DISCRETE_ACTIONS = {0: np.array([-1, 0, 0]),  # steer sharp left
    #                     1: np.array([1, 0, 0]),  # steer sharp right
    #                     2: np.array([-0.5, 0, 0]),  # steer left
    #                     3: np.array([0.5, 0, 0]),  # steer right
    #                     4: np.array([0, 1, 0]),  # accelerate 100%
    #                     5: np.array([0, 0.5, 0]),  # accelerate 50%
    #                     6: np.array([0, 0.25, 0]),  # accelerate 25%
    #                     7: np.array([0, 0, 0.25]),  # brake 25%
    #                     8: np.array([0, 0, 0])}  # do nothing
    DISCRETE_ACTIONS = {0: np.array([-1, 0, 0]),  # steer sharp left
                        1: np.array([1, 0, 0]),  # steer sharp right
                        2: np.array([0, 1, 0]),  # accelerate 100%
                        3: np.array([0, 0, 0.8]),  # brake 25%
                        4: np.array([0, 0, 0])}  # do nothing

    # DISCRETE_ACTIONS = {0: np.array([-1, 0, 0]),  # steer sharp left
    #                     1: np.array([1, 0, 0]),  # steer sharp right
    #                     2: np.array([0, 1, 0]),  # accelerate 100%
    #                     3: np.array([0, 0.5, 0]),  # accelerate 50%
    #                     4: np.array([0, 0, 1]),  # brake 100%
    #                     5: np.array([0, 0, 0.5])}  # brake 50%

    # DISCRETE_ACTIONS = {0: np.array([0, 0, 0]),  # do nothing
    #                     1: np.array([-1, 0, 0]),  # steer sharp left
    #                     2: np.array([1, 0, 0]),  # steer sharp right
    #                     3: np.array([-0.5, 0, 0]),  # steer left
    #                     4: np.array([0.5, 0, 0]),  # steer right
    #                     # 5: np.array([0, 1, 0]),  # accelerate 100%
    #                     # 6: np.array([0, 0.5, 0]),  # accelerate 50%
    #                     # 7: np.array([0, 0.25, 0]),  # accelerate 25%
    #                     # 8: np.array([0, 0, 1]),  # brake 100%
    #                     # 9: np.array([0, 0, 0.5]),  # brake 50%
    #                     # 10: np.array([0, 0, 0.25])}  # brake 25%
    #                     5: np.array([0, 0.5, 0.5]),  # accelerate 50%
    #                     6: np.array([0, 0.25, 0.25]),  # accelerate 50%
    #                     7: np.array([0.5, 0.5, 0]),  # accelerate 50%
    #                     8: np.array([1, 0.5, 0]),  # accelerate 50%
    #                     9: np.array([-1, 0.5, 0]),  # accelerate 50%
    #                     10: np.array([1, 0.25, 0]),  # accelerate 50%
    #                     11: np.array([-1, 0.25, 0]),  # accelerate 50%
    #                     12: np.array([1, 0.5, 0]),  # accelerate 50%
    #                     13: np.array([-1, 0, 0.5]),  # accelerate 50%
    #                     14: np.array([1, 0, 0.5]),  # accelerate 50%
    #                     15: np.array([-1, 0, 0.25]),  # accelerate 50%
    #                     16: np.array([1, 0, 0.25])}  # accelerate 50%


    def __init__(self, environment):
        super(DiscreteCarEnvironment, self).__init__(environment)
        self.action_space = gym.spaces.Discrete(len(self.DISCRETE_ACTIONS))

    def step(self, action):
        return self.env.step(self.DISCRETE_ACTIONS[action])


class EnvironmentWrappers:
    def __init__(self, width, height, num_stacked_frames):
        self.width = width
        self.height = height
        self.num_stacked = num_stacked_frames

    def resize(self, environment):
        return wrappers.ResizeObservation(environment, (self.width, self.height))

    def grayscale(self, environment):
        return wrappers.GrayScaleObservation(environment)

    def frame_stack(self, environment):
        return wrappers.FrameStack(environment, self.num_stacked)

    def observation_wrapper(self, environment, functions):
        return reduce(lambda a, x: x(a), functions, environment)


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Clip reward between -1 to 1
        return np.clip(reward, -1, 1)
    