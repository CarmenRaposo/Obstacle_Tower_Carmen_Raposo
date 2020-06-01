from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime
#import dopamine
import tensorflow as tf
from shutil import copyfile
import time as t
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy, FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec
from stable_baselines.a2c.utils import conv, conv_to_fc, linear, _ln

TEST_STEPS = 2000
TRAINING_INTERVAL_STEPS = 10000
TOTAL_TRAINING_STEPS = 1e12
RESULTS_PATH = "/home/home/Data/Carmen/py_workspace/ObstacleTower/results/" + datetime.now().strftime(
    "%B-%d-%Y_%H_%M%p")
TRAINING_NAME = "dqn_train_my_cnn"
AGENT_ALGORITHM = "DQN"  # DDPG, PPO2, TRPO, DQN
PRETRAINED_MODEL = "/home/home/Data/Carmen/py_workspace/ObstacleTower/results/July-30-2019_21_36PM_dqn_train_my_cnn/dqn_train_my_cnn_0000290000.pkl"
#PRETRAINED_MODEL = None
TEST_ONLY = True
NEURONAL_NETWORK = "CNN"  # CNN, MLP

config = {"visual-theme": 2, "dense-reward": 1}

env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=True, config=config)
# env.seed(5)
input_shape = env.observation_space.shape
print("OBS", env.observation_space)
#print("OBS", env.observation_space[0])
print("INPUT SHAPE", input_shape)
env = DummyVecEnv([lambda: env])


# Create global experiments path
if not TEST_ONLY:
    global_path = RESULTS_PATH + "_" + TRAINING_NAME + "/"
else:
    global_path = RESULTS_PATH + "_" + TRAINING_NAME + "_test" + "/"

os.makedirs(global_path, exist_ok=True)

# Copy file to results directory
if PRETRAINED_MODEL:
    print("PRETRAINED MODEL")
    pretrained_model_name = "pretrained.pkl"
    copyfile(PRETRAINED_MODEL, global_path + pretrained_model_name)

# Define custom policy

print('HOLAAAA', env.observation_space.shape)

if NEURONAL_NETWORK == "CNN":

    from keras import models
    from keras import layers


    def my_cnn(scaled_images, **kwargs):
        """
        CNN from Nature paper.

        :param scaled_images: (TensorFlow Tensor) Image input placeholder
        :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
        :return: (TensorFlow Tensor) The CNN output layer
        """

        #Configuración inicial 32, 64, 64 -> Pruebo con 64, 256, 256
        activ = tf.nn.relu
        layer_1 = activ(
            conv(scaled_images, 'c1', n_filters=64, filter_size=8, stride=4, pad='SAME', init_scale=np.sqrt(2), **kwargs))
        #layer_1_norm = _ln(layer_1)
        layer_2 = activ(conv(layer_1, 'c2', n_filters=256, filter_size=4, stride=2, pad='SAME', init_scale=np.sqrt(2), **kwargs))
        #layer_2_norm = _ln(layer_2)
        layer_3 = activ(conv(layer_2, 'c3', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
        layer_3 = conv_to_fc(layer_3)
        return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

    # NEXT STEP: 3 capas de 128 (anterior 32, 64, 64)
    """convnet = models.Sequential()
    convnet.add(layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    convnet.add(layers.MaxPooling2D((2, 2)))
    convnet.add(layers.Conv2D(64, (3, 3), activation='relu'))
    convnet.add(layers.MaxPooling2D((2, 2)))
    convnet.add(layers.Conv2D(64, (3, 3), activation='relu'))
    convnet.add(layers.Flatten())
    convnet.add(layers.Dense(64, activation='relu'))
    convnet.add(layers.Dense(10, activation='softmax'))"""


    """class CustomPolicy(FeedForwardPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
            super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

            with tf.variable_scope("model", reuse=reuse):
                flat = tf.keras.layers.Flatten()(self.processed_obs)

                x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name='pi_fc_0')(flat)
                pi_latent = tf.keras.layers.Dense(64, activation="relu", name='pi_fc_1')(x)

                x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name='vf_fc_0')(flat)
                vf_latent = tf.keras.layers.Dense(64, activation="relu", name='vf_fc_1')(x1)

                value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)

                self.proba_distribution, self.policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

            self.value_fn = value_fn
            self.initial_state = None
            self._setup_init()

    """
    # CNN Policy with 3 layers of 64, 256 and 256 size each one, respectively
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layer_norm=True,
                                               cnn_extractor=my_cnn,
                                               feature_extraction="cnn")


else:
    # CNN Policy with 3 layers of 256 size each one, includes normalitazion due to the param noise
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[256, 256, 256],
                                               act_fun=tf.nn.relu,
                                               feature_extraction="mlp")
register_policy('CustomPolicy', CustomPolicy)

# Define model
if AGENT_ALGORITHM == "DQN":
    # Add some param noise for exploration
    #param_noise = None
    action_noise = None
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01)

    # Because we use parameter noise, we should use a MlpPolicy with layer normalization
    model = DQN(policy="CustomPolicy", env=env, param_noise=param_noise, verbose=1, tensorboard_log=global_path + "tb")

    # Load if pretrained
    if PRETRAINED_MODEL:
        del model
        model = DQN.load(global_path + pretrained_model_name, policy=CustomPolicy, env=env)
        print("INFO: Loaded model " + global_path + pretrained_model_name)


def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)

    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


# Step counter initialization
t = 0

# Main loop
while (t < TOTAL_TRAINING_STEPS):

    if not TEST_ONLY:
        # Train model
        if (t == 0):
            model.learn(total_timesteps=TRAINING_INTERVAL_STEPS, tensorboard_log='./tensorboard-logs')
        else:
            model.learn(total_timesteps=TRAINING_INTERVAL_STEPS, reset_num_timesteps=False)

    # Evaluate model
    print("Testing model...")
    evaluate(model, num_steps=TEST_STEPS)

    if not TEST_ONLY:
        # Saving model
        print("Saving in '" + global_path + "'")
        model.save(global_path + TRAINING_NAME + "_" + str(int(t)).zfill(10))

    # Update t
    t = t + TRAINING_INTERVAL_STEPS
