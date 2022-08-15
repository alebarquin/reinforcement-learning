import gym
from stable_baselines3 import TD3
import os

models_dir = 'models/TD3'
logs_dir = 'logs/TD3'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = gym.make('Pendulum-v1')

model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)

timesteps = 10000
for i in range(1, 30):
    
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name='TD3')
    model.save(f'{models_dir}/{timesteps*i}')

env.close()