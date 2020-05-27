from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from utils import otc_arg_parser, log
from environment_preprocessing import OTCPreprocessing
from stable_baselines.common.vec_env import DummyVecEnv
from plotter import test
import os

"""TOTAL_TIMESTEPS = 5000000
TRAINING_INTERVAL_STEPS = 1000
RESULTS_PATH = "./results/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p")
TRAINING_NAME = "ppo_v3"
"""



def main():
    #Load parse parameters
    parser = otc_arg_parser()
    args = parser.parse_args()

    global_path = args.results_dir + "_" + args.training_name + "/"

    #Challenge environment
    env = ObstacleTowerEnv('/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/ObstacleTower-v3.1/obstacletower.x86_64', retro=args.retro, realtime_mode=args.realtime)

    #Dict of actions created by the ObstacleTowerEnv Class of obstacle_tower_env library
    #print("ACTIONS:", env._flattener.action_lookup)
    
    #Preprocess the environment (Grey Scales and action space reduction)
    env = OTCPreprocessing(env, args.action_reduction)
    env = DummyVecEnv([lambda: env])

    #env = make_vec_env(env, n_envs=4)

    ########Training########

    if args.use_gae:

        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=args.tensorboard_logdir,
                            cliprange=args.clip_param, learning_rate=args.lr, ent_coef=args.entropy_coef,
                            vf_coef=args.value_loss_coef, max_grad_norm=args.max_grad_norm,
                            gamma=args.gamma, lam=args.gae_lambda)
    else:

        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=args.tensorboard_logdir,
                            cliprange=args.clip_param, learning_rate=args.lr, ent_coef=args.entropy_coef,
                            vf_coef=args.value_loss_coef, max_grad_norm=args.max_grad_norm,
                            gamma=args.gamma)

    #model.learn(total_timesteps=50000)
    #model.save("ObstacleTower_prueba")

    t=0
    while t < args.num_env_steps:
        #TRAIN MODEL
        if t == 0:
            model.learn(total_timesteps=args.eval_interval)

        else:
            model.learn(total_timesteps=args.eval_interval, reset_num_timesteps=False)

        os.makedirs(global_path, exist_ok=True)
        print("Saving in '" + global_path + "'")
        model.save(global_path + args.training_name + "_" + str(int(t)).zfill(10))

        avg_reward, avg_floor = test(t, model, env=env)  # Test
        log('T = ' + str(t) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(
            avg_reward) + ' | Avg. floor: ' + str(avg_floor))

        t += args.eval_interval
"""
    obs = env.reset()
    while True:
        action, _states = model.prediction(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
"""

if __name__ == "__main__":
    main()


