from datetime import datetime

def otc_arg_parser():

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--action_reduction',
        action='store_true',
        default=True, #False
        help='reduce the action space dimensionality')
    parser.add_argument(
        '--image_classifier',
        action='store_true',
        default=False, #True
        help='classify the features recognized in the visual input')
    parser.add_argument(
        '--retro',
        default=True,#False
        help='retro mode parameter for the env')
    parser.add_argument(
        '--realtime',
        default=False,#True
        help='realtime mode parameter for the env')
    parser.add_argument(
        '--lr',
        type=float,
        # default=7e-4,
        # default=2.5e-4,
        default=1e-4,
        help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        # default=0.96,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True, # False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        # default=0.01,
        default=0.001,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default =0.5,
        # default=1.,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        # default=16,
        default=32, # 16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        # default=5,
        default=512,
        # default=256,
        # default=128,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        # default=4,
        default=8,
        # default=3,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        #=8,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        # default=0.2,
        default=0.1,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        # default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10000,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env',
        default='ObtRetro-v6', #'PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='./results/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True, # False,
        # default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        # default=True,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--tensorboard-logdir',
        default = "./results/tensorboard/logs_tensorboard" + datetime.now().strftime("%B-%d-%Y_%H_%M%p"),
        help = 'dir of the tensorboard logs'
    )
    parser.add_argument(
        '--results-dir',
        default="./results/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p"),
        help="dir of the results evaluation logs")

    parser.add_argument(
        '--training-name',
        default="ppo_v3",
        help="name of the training saved file"
    )

    parser.add_argument(
        '--policy',
        help='Policy architecture',
        choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
        default='lnlstm')



    # parser.add_argument('--lbda', type=float, default=0.95)
    # parser.add_argument('--gamma', type=float, default=0.96)  # 0.99
    # parser.add_argument('--nminibatches', type=int, default=8)
    # parser.add_argument('--norm_adv', type=int, default=1)
    # parser.add_argument('--norm_rew', type=int, default=0)
    # parser.add_argument(
    #     '--lr', type=float, default=1e-4)  # lambda f: f * 2.5e-4,
    # parser.add_argument('--ent_coeff', type=float, default=0.001)
    # parser.add_argument('--nepochs', type=int, default=8)
    parser.add_argument('--nsteps_per_seg', type=int, default=512)
    parser.add_argument('--num_practice_agents', type=int, default=0)
    # parser.add_argument('--num_timesteps', type=int, default=int(1e7))

    # parser.add_argument('--env', help='environment ID', default='ObtRetro-v4')
    # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument(
        'environment_filename',
        default='../../../ObstacleTower/obstacletower',
        nargs='?')
    # parser.add_argument('--envs_per_process', type=int, default=8)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--score', type=bool, default=False)
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    parser.add_argument('--sample_normal', action='store_true')
    parser.add_argument('--seed_from_10', action='store_true')
    parser.add_argument('--alfie', action='store_true')
    parser.add_argument('--level_is_rand', action='store_true')
    parser.add_argument('--exp_name', type=str, default='debug')
    # parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--load', action='store_true')
    # parser.add_argument('--normalize_visual_obs', action='store_true')
    parser.add_argument('--inverse_rl', action='store_true')
    parser.add_argument('--action_set_5', action='store_true')
    parser.add_argument('--action_set_6', action='store_true')
    parser.add_argument('--action_set_20', action='store_true')
    parser.add_argument('--action_set_27', action='store_true')
    parser.add_argument('--action_set_54', action='store_true')
    parser.add_argument('--half_precision', action='store_true')

    return parser

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

def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)