import gym
from gym import spaces
from features_classifier import StateFeatures, StateClassifier
import torch
import numpy as np
from rnd_agent import *
import pdb

class OTCPreprocessing(gym.Wrapper):
    """A class implementing image preprocessing for OTC agents.

    Specifically, this converts observations to greyscale. It doesn't
    do anything else to the environment.
    """

    def __init__(self, environment, action_reduction, features):
        """Constructor for an Obstacle Tower preprocessor.

        Args:
            environment: Gym environment whose observations are preprocessed.

        """
        self.env = environment

        environment.action_meanings = ['NOOP']

        self.game_over = False
        #self.lives = 0  # Will need to be set by reset().
        self.action_reduction = action_reduction
        self.features = features
        self.target_feature_batch = None
        self.predict_feature_batch = None
        self.n_step = 0

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
    def observation_space(self):
        if self.features:
            image_space = spaces.Box(0, 255, dtype=np.uint8, shape=(84, 84, 3))
            #return spaces.Box(low=-256, high=256, shape=(21168, 1), dtype=np.float32)
            #return spaces.Discrete(21179)
            obs_space = spaces.Box(0, 255, dtype=np.uint8, shape=(21179,))
            return obs_space
            #return spaces.Tuple((image_space, spaces.Discrete(11)))
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

        if self.features:
            device = torch.device('cuda')
            clasificador = StateClassifier()
            clasificador.load_state_dict(torch.load('save_classifier.pkl', map_location='cpu'))
            clasificador.to(device)
            #class_observation = np.array(observation)
            class_observation = torch.from_numpy(np.array(observation)[None]).to(device)
            features = clasificador(class_observation).detach().cpu().numpy()[0]#Change the boolean array to a binary array (length 11)
            features = features > 0
            features = [0 if features[i] is False else 255 for i in range(len(features))]#White pixel if the feature is; black if it's not
            #print('features :', features)
            observation = observation.reshape(21168) #Reshape the observation to manage properly the observation space
            #features = features.reshape(11,1)
            #pdb.set_trace()
            observation = np.append(observation, features) #Add the features array as part of the observation
            #observation = observation[:, 0]
            # print('observation : ', observation[:])
            # print('observation shape and type', observation.shape, observation.dtype)

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

        self.n_step += 1
        #print('action : ', action)
        #print('actionInput : ', actionInput)

        observation, extrinsic_reward, game_over, info = self.env.step(actionInput)
        self.game_over = game_over
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        info['actual_action'] = actionInput
        info['actual_inner_action'] = action

        # print('info', info)
        # print('info[current_floor]', info['current_floor'])

        if self.features:
            device = torch.device('cuda')
            clasificador = StateClassifier()
            clasificador.load_state_dict(torch.load('save_classifier.pkl', map_location='cpu'))
            clasificador.to(device)
            #class_observation = np.array(observation)
            class_observation = torch.from_numpy(np.array(observation)[None]).to(device)
            features = clasificador(class_observation).detach().cpu().numpy()[0]#Change the boolean array to a binary array (length 11)
            features = features > 0
            features = [0 if features[i] is False else 255 for i in range(len(features))] #White pixel if the feature is; black if it's not
            #print('features :', features)


        if args.rnd:
            agent = RNDAgent()
            rnd_observation = np.moveaxis(observation, 2, 0)
            #print('rnd observation shape 1:', rnd_observation.shape)
            rnd_observation = np.array([rnd_observation])
            #print('rnd observation shape 2:', rnd_observation.shape)
            predict_feature, target_feature = agent.rnd.forward(rnd_observation)
            predict_feature, target_feature, intrinsic_reward = agent.intrinsic_reward(rnd_observation)
            #predict_feature, target_feature = predict_feature.tolist(), target_feature.tolist()
            intrinsic_reward = intrinsic_reward.tolist()
            #print('nstep:', self.n_step)
            cond = self.target_feature_batch
            if cond is None:
                self.target_feature_batch = [target_feature]
                self.predict_feature_batch = [predict_feature]
                #print('List is empty')
                #pass #Implement the error

            else:
                #print('Entra en el batch')
                self.target_feature_batch.append(target_feature)
                self.predict_feature_batch.append(predict_feature)
                #print('Batchs actualizados: ', self.target_feature_batch, self.predict_feature_batch)
                # print('Shapes of the lists: ', np.array(self.predict_feature_batch).shape,
                #       np.array(self.target_feature_batch).shape)

            if (self.n_step % args.rnd_batch_size) == 0:
                #print('Training rnd step :', self.n_step)
                agent.train_rnd(self.predict_feature_batch, self.target_feature_batch)
                self.predict_feature_batch = []
                self.target_feature_batch = []


            #print('extrinsic and intrisic reward: ', extrinsic_reward, intrinsic_reward[0])
            reward = extrinsic_reward + intrinsic_reward[0]
            #print('reward :', reward)

        else:
            reward = extrinsic_reward

        observation = observation.reshape(21168)  # Reshape the observation to manage properly the observation space
        # features = features.reshape (11,1)
        # pdb.set_trace()
        observation = np.append(observation, features)  # Add the features array as part of the observation
        # observation = observation[:, 0]
        # print('observation : ', observation[:])
        # print('observation shape', observation.shape)

        return observation, reward, game_over, info





        # def unwrap(self):
    #     if hasattr(self.env, "unwrapped"):
    #         return env.unwrapped
    #     elif hasattr(self.env, "env"):
    #         return unwrap(self.env.env)
    #     elif hasattr(self.env, "leg_env"):
    #         self.unwrap = unwrap(self.env.leg_env)
    #         return self.unwrap
    #     else:
    #         return self.env

