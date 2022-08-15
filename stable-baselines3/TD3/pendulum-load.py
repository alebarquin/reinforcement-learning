import gym
from stable_baselines3 import TD3
import os

models_dir = 'models/TD3'
logs_dir = 'logs/TD3'
model_path = f'{models_dir}/40000.zip'

env = gym.make('Pendulum-v1')

model = TD3.load(model_path, env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()