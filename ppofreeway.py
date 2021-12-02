import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("FreewayNoFrameskip-v0")

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("FreewayNoFrameskip-v0")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
