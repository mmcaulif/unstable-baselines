import gym

from stable_baselines3 import TD3

env = gym.make('Pendulum-v1')

model = TD3('MlpPolicy', env, batch_size=100, verbose=1)

model.learn(total_timesteps=300000)