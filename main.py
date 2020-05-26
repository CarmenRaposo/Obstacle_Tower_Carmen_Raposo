from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from utils import otc_arg_parser
from environment_preprocessing import OTCPreprocessing
from stable_baselines.common.vec_env import DummyVecEnv
#import argparse

def main():
    #Load parse parameters
    parser = otc_arg_parser()
    args = parser.parse_args()

    #print("args : ", args)

    #Challenge environment
    env = ObstacleTowerEnv('/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/ObstacleTower-v3.1/obstacletower.x86_64', retro=args.retro, realtime_mode=args.realtime)

    print("ACTIONS:", env._flattener.action_lookup)
    #Preprocess the environment (Grey Scales and action space reduction)
    env = OTCPreprocessing(env, args.action_reduction)
    env = DummyVecEnv([lambda: env])

    #env = make_vec_env(env, n_envs=4)

    ########Training########
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ObstacleTower_prueba")



"""
    obs = env.reset()
    while True:
        action, _states = model.prediction(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
"""

if __name__ == "__main__":
    main()
