import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("FreewayNoFrameskip-v0", n_envs=8)

model = PPO("CnnPolicy", env, verbose=1,
    learning_rate=lambda x: 2.5*1e-4*x, 
    n_steps=128, 
    batch_size=32*8, 
    n_epochs=3,
    clip_range=lambda x: 0.1*x, 
    ent_coef=0.001, 
    vf_coef=1, 
)
model.learn(total_timesteps=1e6)
model.save("FreewayNoFrameskip-v0")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
