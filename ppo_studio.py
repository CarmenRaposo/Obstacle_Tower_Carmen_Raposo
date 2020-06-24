from constants import *
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import os
from plotter import test
from utils import log
import time
import random


def params_test(policy, env, n_steps=args.n_steps_test, ppo_epochs=args.ppo_epoch_test, clip_params=args.clip_param_test,
                gammas=args.gamma_test, lambdas=args.gae_lambda_test, loss_coefs=args.value_loss_coef_test,
                entropy_coefs=args.entropy_coef_test, lrs=args.lr_test):

    i = 0
    for n_step in n_steps: #[512]
        for ppo_epoch in ppo_epochs: #[4]
            for clip_param in clip_params: #[0.1, 0.2]
                for gamma in gammas: #[0.99, 0.9997]
                    for lb in lambdas: #[0.95]
                        for value_loss_coef in loss_coefs: #[0.5]
                            for entropy_coef in entropy_coefs: #[0.01, 0.001]
                                for lr in lrs: #[2.5e-4, 7e-4]

                                    i += 1

                                    # if i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16]: #Falta el 6 y el 14 por hacer completo
                                    #     pass
                                    # else:

                                        # if i == 15:
                                        #     print('Continue Training')
                                        #     trained_model = "/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/python_scripts/Obstacle_Tower_Carmen_Raposo/results/June-17-2020_01_46AM/model/__15__study_0000350000.zip"
                                        #     model = PPO2.load(trained_model, env=env, tensorboard_log="/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/python_scripts/Obstacle_Tower_Carmen_Raposo/results/June-17-2020_01_46AM/tensorboard/15/")
                                        #     t = 375000
                                        #     GLOBAL_PATH = "/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/python_scripts/Obstacle_Tower_Carmen_Raposo/results/June-17-2020_01_46AM/model/"
                                        #     filename = 'argsparams' + str(i) + '.txt'
                                        #     os.makedirs(args.results_dir, exist_ok=True)
                                        #
                                        #
                                        # else:
                                    print('Start Training: \n n_step: %f \n ppo_epoch: %f \n clip_param: %f \n gamma: %f'
                                          '\n lambda: %f \n value_loss_coef: %f \n entropy_coef: %f \n learning_rate : %f'
                                          % (n_step, ppo_epoch, clip_param, gamma, lb, value_loss_coef, entropy_coef,
                                             lr))
                                    #Fixed seed
                                    seed = random.seed(0)
                                    #env.seed(5)

                                    if args.use_gae_test:

                                        model = PPO2(policy, env, n_steps=n_step, verbose=1, tensorboard_log=args.tensorboard_logdir + str(i) + '/',
                                                     cliprange=clip_param, learning_rate=lr, ent_coef=entropy_coef,
                                                     vf_coef=value_loss_coef, max_grad_norm=args.max_grad_norm,
                                                     gamma=gamma, lam=lb, noptepochs=ppo_epoch, seed=seed)
                                    else:

                                        model = PPO2(policy, env, n_steps=n_step, verbose=1, tensorboard_log=args.tensorboard_logdir + str(i) + '/',
                                                     cliprange=clip_param, learning_rate=lr, ent_coef=entropy_coef,
                                                     vf_coef=value_loss_coef, max_grad_norm=args.max_grad_norm,
                                                     gamma=gamma, noptepochs=ppo_epoch, seed=seed)


                                    #Save the values of the configured parameters
                                    filename = 'argsparams' + str(i) + '.txt'
                                    os.makedirs(args.results_dir, exist_ok=True)
                                    myfile = open(args.results_dir + filename, 'w+')
                                    myfile.write('n_step: %f \n ppo_epoch: %f \n clip_param: %f \n gamma: %f \n lambda: %f ' 
                                             '\n value_loss_coef: %f \n entropy_coef: %f \n learning_rate : %f'
                                             % (n_step, ppo_epoch, clip_param, gamma, lb, value_loss_coef, entropy_coef,
                                                lr))
                                    myfile.close()
                                    t = 0

                                    #t = 0
                                    while t < args.num_env_steps_test:
                                        # TRAIN MODEL
                                        try:
                                            if t == 0:
                                                model.learn(total_timesteps=args.eval_interval)

                                            else:
                                                model.learn(total_timesteps=args.eval_interval,
                                                            reset_num_timesteps=False)

                                            os.makedirs(GLOBAL_PATH, exist_ok=True)
                                            print("Saving in '" + GLOBAL_PATH + "'")
                                            model.save(GLOBAL_PATH + '__' + str(i) + '__' + args.training_name + "_" + str(int(t)).zfill(10))

                                            avg_reward, avg_floor = test(t, model, env=env,
                                                                         global_path=GLOBAL_PATH + '__' + str(i), i=i)  # Test
                                            log('T = ' + str(t) + ' / ' + str(
                                                args.num_env_steps_test) + ' | Avg. reward: ' + str(
                                                avg_reward) + ' | Avg. floor: ' + str(avg_floor))

                                            t += args.eval_interval
                                        except Exception as e:

                                            env.close()

                                            myfile = open(GLOBAL_PATH + filename, 'a')
                                            myfile.write('\n An exception %s has occured at step %f' % (e, t))
                                            myfile.close()

                                            del model

                                            from obstacle_tower_env import ObstacleTowerEnv
                                            env = ObstacleTowerEnv('/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/ObstacleTower-v3.1/obstacletower.x86_64',
                                                  retro=args.retro, realtime_mode=args.test, timeout_wait=6000)

                                            break

                                    env.reset()
                                    break
                                    del model
                                    #time.sleep(300)







