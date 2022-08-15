import gym
from stable_baselines3 import SAC
import os

models_dir = 'models/SAC'
logs_dir = 'logs/SAC'
model_path = f'{models_dir}/90000.zip'

env = gym.make('Pendulum-v1')

model = SAC.load(model_path, env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()