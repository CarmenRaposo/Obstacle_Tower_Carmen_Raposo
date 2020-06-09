import gym
from gym import spaces
import numpy as np
import cv2

class OTCPreprocessing(gym.Wrapper):
    """A class implementing image preprocessing for OTC agents.

    Specifically, this converts observations to greyscale. It doesn't
    do anything else to the environment.
    """

    def __init__(self, environment, action_reduction):
        """Constructor for an Obstacle Tower preprocessor.

        Args:
            environment: Gym environment whose observations are preprocessed.

        """
        self.env = environment

        environment.action_meanings = ['NOOP']

        self.game_over = False
        #self.lives = 0  # Will need to be set by reset().
        self.action_reduction = action_reduction

        if action_reduction:
                # Reduction of the action space dimensionality wit only 8 basic possible movements
                """self.actions = {
                    0: [0, 0, 0, 0],  # nop
                    18: [1, 0, 0, 0],  # forward
                    36: [2, 0, 0, 0],  # backward
                    6: [0, 1, 0, 0],  # cam left
                    12: [0, 2, 0, 0],  # cam right
                    21: [1, 0, 1, 0],  # jump forward
                    24: [1, 1, 0, 0],  # forward + cam left
                    30: [1, 2, 0, 0]  # forward + cam right
                }"""

                self.actions = {
                    0: [0, 0, 0, 0],  # nop
                    1: [1, 0, 0, 0],  # forward
                    2: [2, 0, 0, 0],  # backward
                    3: [0, 1, 0, 0],  # cam left
                    4: [0, 2, 0, 0],  # cam right
                    5: [1, 0, 1, 0],  # jump forward
                    6: [1, 1, 0, 0],  # forward + cam left
                    7: [1, 2, 0, 0]  # forward + cam right
                }




    @property
    def observation_space(self, image_classifier=False):
        if image_classifier:
            pass
        else:
            return self.env.observation_space

    @property
    def action_space(self):
        if self.action_reduction:
            return spaces.Discrete(len(self.actions))
        else:
            return self.env.action_space

        # return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    def reset(self):
        """Resets the environment. Converts the observation to greyscale,
        if it is not.

        Returns:
        observation: numpy array, the initial observation emitted by the
            environment.
        """
        observation = self.env.reset()
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        return observation

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
        mode: Mode argument for the environment's render() method.
            Valid values (str) are:
            'rgb_array': returns the raw ALE image.
            'human': renders to display via the Gym renderer.

        Returns:
        if mode='rgb_array': numpy array, the most recent screen.
        if mode='human': bool, whether the rendering was successful.
        """
        return self.env.render(mode)

    def step(self, actionInput):
        """Applies the given action in the environment. Converts the observation to
        greyscale, if it is not.

        Remarks:

        * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
        action: The action to be executed.

        Returns:
        observation: numpy array, the observation following the action.
        reward: float, the reward following the action.
        is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
        info: Gym API's info data structure.
        """
        # ['Movement Forward/Back', 'Camera', 'Jump', 'Movement Left/Right']
        # [3, 3, 2, 3]
        from typing import Iterable
        if isinstance(actionInput, Iterable):
            action = self.actions[actionInput[0]]
        else:
            action = self.actions[actionInput]


        #print('action : ', action)
        #print('actionInput : ', actionInput)

        observation, reward, game_over, info = self.env.step(actionInput)
        self.game_over = game_over
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        info['actual_action'] = actionInput
        info['actual_inner_action'] = action
        return observation, reward, game_over, info

    def unwrap(self):
        if hasattr(self.env, "unwrapped"):
            return env.unwrapped
        elif hasattr(self.env, "env"):
            return unwrap(self.env.env)
        elif hasattr(self.env, "leg_env"):
            self.unwrap = unwrap(self.env.leg_env)
            return self.unwrap
        else:
            return self.env
