import gym
from stable_baselines3 import PPO
import os

models_dir = 'models/PPO'
logs_dir = 'logs/PPO'
model_path = f'{models_dir}/190000.zip'

env = gym.make('Pendulum-v1')

model = PPO.load(model_path, env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()