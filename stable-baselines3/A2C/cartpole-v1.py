import gym
from stable_baselines3 import A2C
import os

models_dir = 'models/A2C'
logs_dir = 'logs/A2C'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)

timesteps = 10000
for i in range(1, 30):
    
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name='A2C')
    model.save(f'{models_dir}/{timesteps*i}')

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()

env.close()