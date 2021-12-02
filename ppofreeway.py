import sys
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
#TODO: Add Atari Wrapper (see stable baselines make_atari_env)
env = make_vec_env("FreewayNoFrameskip-v0", n_envs=8)

def train_model(n_iters):
    model = PPO("CnnPolicy", env, verbose=1,
        learning_rate=lambda x: 2.5*1e-4*x, 
        n_steps=128, 
        batch_size=32*8, 
        n_epochs=3,
        clip_range=lambda x: 0.1*x, 
        ent_coef=0.001, 
        vf_coef=1, 
        device="cpu",
        tensorboard_log='logs/'
    )

    model.learn(total_timesteps=n_iters)
    model.save(f"FreewayNoFrameskip-v0-n_steps{128}-batch_size{32*8}-timesteps{n_iters}")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':
    n_iters = eval(sys.argv[1])
    print(f"n iters: {n_iters}")
    train_model(n_iters)
