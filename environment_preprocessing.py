import gym
from gym import spaces
import numpy as np
import cv2


class RetroWrapper(gym.ObservationWrapper):

    def __init__(self, env, randomize, size=84, keep_obs=False):
        super(RetroWrapper, self).__init__(env)
        self._randomize = randomize
        self._size = size
        self._is_otc = hasattr(self.observation_space, 'spaces')
        depth = 3
        self._8bit = True
        # self._8bit = False
        self._keep_obs = keep_obs
        # if not self._keep_obs:
        image_space_max = 1.0
        image_space_dtype = np.float32
        if self._8bit:
            image_space_max = 255
            # image_space_dtype = np.uint8
        camera_height = size
        camera_width = size

        image_space = spaces.Box(
            0, image_space_max,
            dtype=image_space_dtype,
            shape=(camera_height, camera_width, depth)
        )
        if self._is_otc:
            self._spaces = (image_space, self.observation_space[1], self.observation_space[2])
            self._vector_obs_size = self.observation_space[1].n + self.observation_space[2].shape[0]
            self._vector_time_idx = self.observation_space[1].n
            vector_obs_shape = spaces.Box(0, 1., dtype=np.float32, shape=([self._vector_obs_size]))
            self.observation_space = spaces.Dict({'visual': image_space, 'vector': vector_obs_shape})
        else:
            self.observation_space = image_space

    def _get_vector_obs(self):
        if self._is_otc:
            v = np.zeros(self._vector_obs_size, dtype=np.float32)
            v[self._key] = 1
            v[self._vector_time_idx] = self._time
            return v
        return None

    def observation(self, obs):
        w = self._size
        h = self._size
        # extract observations
        hd_visual_obs = obs[0]
        key = obs[1]
        time = obs[2]
        if self._randomize:
            if np.random.choice([0,1]):
                key = 5
                time = 10000
            else:
                key = 0
                time = 0
        self._key = key
        print("time", time)
        self._time = min(time.any(), 10000) / 10000


        print("Hd visual ", hd_visual_obs)
        print("Hd visual shape", hd_visual_obs.shape)

        if hd_visual_obs.shape == self.observation_space.shape:
            print("Entra en hd")
            visual_obs = hd_visual_obs
        else:
            print("Entra en el else")
            # resize
            # from PIL import Image
            # # hd_visual_obs = (255.0 * hd_visual_obs).astype(np.uint8)
            # obs_image = Image.fromarray(hd_visual_obs)
            # obs_image = obs_image.resize((84, 84), Image.NEAREST)
            # visual_obs = np.array(obs_image)
            # # visual_obs = (visual_obs).astype(np.float32) / 255.
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_AREA)
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
            visual_obs = np.array(obs_image)

        if not self._keep_obs:
            # Displays time left and number of keys on visual observation
            # key = vector_obs[0:6]
            # time = vector_obs[6]
            # key_num = np.argmax(key, axis=0)
            key_num = self._key
            time_num = self._time

            max_bright = 1

            print("visual_obs shape :", visual_obs.shape)

            visual_obs[0:10, :, :] = 0
            for i in range(key_num):
                start = int(i * 16.8) + 4
                end = start + 10
                visual_obs[1:5, start:end, 0:2] = max_bright
            visual_obs[6:10, 0:int(time_num * w), 1] = max_bright

        if self._8bit:
            # visual_obs = (255.0 * visual_obs).astype(np.uint8)
            visual_obs = (255.0 * visual_obs)
        # else:
        #     visual_obs = (255.0 * visual_obs)
        if self._is_otc:
            v = {'visual': visual_obs, 'vector': self._get_vector_obs()}
            return v
        return visual_obs


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
