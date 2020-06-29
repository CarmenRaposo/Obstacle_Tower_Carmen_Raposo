from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from utils import otc_arg_parser, log
from environment_preprocessing import OTCPreprocessing
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from plotter import test
import os
from constants import *
from ppo_studio import params_test
import random

"""TOTAL_TIMESTEPS = 5000000
TRAINING_INTERVAL_STEPS = 1000
RESULTS_PATH = "./results/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p")
TRAINING_NAME = "ppo_v3"
"""


def main():
    #Load parse parameters
    #parser = otc_arg_parser()
    #args = parser.parse_args()


    #Challenge environment
    # if args.env == 'ObtRetro-v6':
    #     env = ObstacleTowerEnv(
    #         '/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/ObstacleTower-v3.1/obstacletower.x86_64',
    #         timeout_wait=6000,
    #         retro=args.retro,
    #         realtime_mode=args.test)
    #     env = RetroWrapper(env, args.sample_normal)
    #     env = OTCPreprocessing(env, args.action_reduction)
    #     # if show_obs:
    #     #     env = RenderObservations(env)
    #     #     env = KeyboardControlWrapper(env)
    # else:
    env = ObstacleTowerEnv('/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/ObstacleTower-v3.1/obstacletower.x86_64',
                         retro=args.retro, realtime_mode=args.test, timeout_wait=6000)

    #env = ObstacleTowerEnv('OBSTACLE_TOWER_PATH', retro=args.retro, realtime_mode=args.test, timeout_wait=6000)

    #Dict of actions created by the ObstacleTowerEnv Class of obstacle_tower_env library
    #print("ACTIONS:", env._flattener.action_lookup)


    print('FEATURES :', args.features)

    #Preprocess the environment (Grey Scales and action space reduction)
    env = OTCPreprocessing(env, args.action_reduction, args.features)
    env = DummyVecEnv([lambda: env])
    #env = VecEnv(1, env.observation_space, env.action_space)

    print("ACTION SPACE  ///////////:", env.action_space)
    print("OBSERVATION SPACE ///////////////:", env.observation_space)
    #env = make_vec_env(env, n_envs=4)

    ########Training########

    #Study of the impact of different values of the PPO params
    if args.study:
        params_test(MlpPolicy, env)

    #If no Study Mode
    else:
        #If no Test Mode
        if not args.test:

            seed = random.seed(0)

            #If Generalized Advantage Estimator is used
            if args.use_gae:

                model = PPO2(MlpPolicy, env, n_steps=args.num_steps, verbose=1, tensorboard_log=args.tensorboard_logdir,
                             cliprange=args.clip_param, learning_rate=args.lr, ent_coef=args.entropy_coef,
                             vf_coef=args.value_loss_coef, max_grad_norm=args.max_grad_norm,
                             gamma=args.gamma, lam=args.gae_lambda, noptepochs=args.ppo_epoch, seed=seed)

            #If Generalized Advantage Estimator is not used
            else:

                model = PPO2(MlpPolicy, env, n_steps=args.num_steps, verbose=1, tensorboard_log=args.tensorboard_logdir,
                             cliprange=args.clip_param, learning_rate=args.lr, ent_coef=args.entropy_coef,
                             vf_coef=args.value_loss_coef, max_grad_norm=args.max_grad_norm,
                             gamma=args.gamma, noptepochs=args.ppo_epoch, seed=seed)
        else:

            model = PPO2.load(args.pretrained_model, env=env)

        #model.learn(total_timesteps=50000)
        #model.save("ObstacleTower_prueba")


        filename = 'argsparams.txt'
        os.makedirs(args.results_dir, exist_ok=True)
        myfile = open(args.results_dir + filename, 'a')
        myfile.write('clip range: %f \n learning rate: %f \n coeficiente de entropía: %f \n coeficiente de pérdida: %f \n '
                     'máximo gradiente: %f \n gamma: %f \n ppo epoch: %f \n'
                     % (args.clip_param, args.lr, args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
                        args.gamma, args.ppo_epoch))
        myfile.close()

        if not args.test:
            t = 0
            while t < args.num_env_steps:
                #TRAIN MODEL
                if t == 0:
                    model.learn(total_timesteps=args.eval_interval)

                else:
                    model.learn(total_timesteps=args.eval_interval, reset_num_timesteps=False)

                os.makedirs(GLOBAL_PATH, exist_ok=True)
                print("Saving in '" + GLOBAL_PATH + "'")
                model.save(GLOBAL_PATH + args.training_name + "_" + str(int(t)).zfill(10))

                avg_reward, avg_floor = test(t, model, env=env, global_path=args.results_dir)  # Test
                log('T = ' + str(t) + ' / ' + str(args.num_env_steps) + ' | Avg. reward: ' + str(
                    avg_reward) + ' | Avg. floor: ' + str(avg_floor))

                t += args.eval_interval
        else:
            obs = env.reset()
            t = 0
            while t < args.num_env_steps:

                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                #print('action :', info)
                env.render('rgb_array')


if __name__ == "__main__":
    main()


