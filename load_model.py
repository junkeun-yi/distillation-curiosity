# original : https://github.com/DLR-RM/rl-baselines3-zoo

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import argparse
import os, sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils3.import_envs  # noqa: F401 pylint: disable=unused-import
from utils3 import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

################################################################
# to clear cv2 Import error
ros_pack_path = '/opt/ros/%s/lib/python2.7/dist-packages'%os.getenv('ROS_DISTRO')
if ros_pack_path in sys.path: sys.path.remove(ros_pack_path)
################################################################


def main():
    env_id = 'AntBulletEnv-v0'
    algo = 'td3'
    folder = "rl-trained-agents"
    n_timesteps = 1000


    exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    print(f"Loading latest experiment, id={exp_id}")

    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    set_random_seed(0)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=0,
        log_dir=None,
        should_render=True,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=0)
    # Dummy buffer size as we don't need memory to enjoy the trained agent
    kwargs.update(dict(buffer_size=1))


    model = ALGOS[algo].load(model_path, env=env, **kwargs)

    obs = env.reset()

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    try:
        for _ in range(n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, infos = env.step(action)
            env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if done:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print(f"Episode Reward: {episode_reward:.2f}")
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0
                state = None

            # Reset also when the goal is achieved when using HER
            if done and infos[0].get("is_success") is not None:
                print("Success?", infos[0].get("is_success", False))

                if infos[0].get("is_success") is not None:
                    successes.append(infos[0].get("is_success", False))
                    episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main()