# CS285 Project code

## Requirements

Optionally (and recommended), make a virtual environment.

0. `pip install wheel`
1. `pip install -r requirements.txt`
2. Go to following link for checkpoints. Checkpoints must be added as `{code_dir}/rl_trained_agents/ppo/FreewayNoFrameskip-v0.zip` 
   - https://drive.google.com/drive/folders/1kKigj-4_VqQ02fqRQKN6SdYVnElkwXHk?usp=sharing
3. In order to import ROMS, you need to download Roms.rar from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the .rar file. Once you've done that, run:
   - `python -m atari_py.import_roms <path to folder>`
   - This should print out the names of ROMs as it imports them. The ROMs will be copied to your atari_py installation directory.
4. `ale-import-roms --import-from-pkg atari_py.atari_roms`

To run: \
`python policy_distillation.py --algo ppo --env FreewayNoFrameskip-v0`

HW5 CODEBASE TODOs:
- [ ] Train PPO Teacher w/ Atari Wrapper (and checkpt throughout training) (THURS)
- [ ] Add Teacher into the codebase (THURS)
- [ ] Add distillation loss (THURS/FRI)
- [ ] Figure out curiosity (should we add inverse dynamics, or rnd??) (FRI/SAT)
- [ ] Figure out schedules (SAT-)

Current TODOs:
- [ ] Set random lib seed for memory sampling
- [ ] Fix Env mismatch (PPO script trains without the Atari wrapper found in util2.utils)
- [ ] Get PPO Checkpoints for various rewards
- [ ] Fix Teacher Performance in Distillation (trained to 21 but in policy distill only reaches 10?!?)
- [ ] Fix Training Bug (0 loss!)
- [ ] Switch to Student Data Collection (instead of current teacher based)
- [ ] Get a list of tasks (from Burda paper). Evaluate relevance
- [ ] Add curiosity module to student
- [ ] Add intrinsic reward scheduling













# policy-distillation-baselines

Pytorch Implementation of Policy Distillation for control, which has well-trained teachers via [stable_baselines3](https://github.com/DLR-RM/stable-baselines3).



STATUS : [`DONE`](https://github.com/CUN-bjy/policy-distillation-baselines/projects)



#### Notice

> *This repository is based on [Mee321/policy-distillation](https://github.com/Mee321/policy-distillation) and integrated with [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) environment.*

## Demonstration

#### Trained Agent(left) and Distilled Agent(right), see more [demo](https://github.com/CUN-bjy/policy-distillation-baselines/issues/3#issuecomment-817730173)..

Trained agent uses **400 x 300** parameters, while distilled agent uses **64 x 64** parameters.

And it just takes 6,000 iteration(about 100 seconds on intel i7) to receive 100% of performance.

![teacher_text_cut](./docs/gifs/1_teacher_text_cut.gif)![pd_baselines_compact](./docs/gifs/distilled_agents.gif)



## Overview

![](./docs/pd_baselines_figures-Page-2.svg)

[**`ZOOM In`**](https://raw.githubusercontent.com/CUN-bjy/policy-distillation-baselines/main/docs/pd_baselines_figures-Page-2.svg)



## Installation

```bash
git clone https://github.com/CUN-bjy/policy-distillation-baselines.git
cd policy-distillation-baselines
git submodule update --init
virtualenv venv
source venv/bin/active
venv/bin/pip install -r requirements.txt
```

You don't need to use virtual environment but recommended.

With every moment of using this package, you should source the `venv`. plz  `source venv/bin/active`.



## Play a Trained Agent

If you want to play trained_agent from stable_baselines3,

```bash
python playground.py --mode teacher --algo algo_name --env env_name
# For example,
# python playground.py --mode teacher --algo td3 --env AntBulletEnv-v0 (default)
# python playground.py --mode teacher --algo sac --env Pendulum-v0
```

See the details below!

```bash
usage: playground.py [-h] -m {teacher,student} [--env ENV]
                     [--algo {a2c,ddpg,dqn,ppo,her,sac,td3,qrdqn,tqc}]
                     [-f FOLDER] [-p PATH_TO_STUDENT] [--render RENDER]
                     [--testing-batch-size N]

optional arguments:
  -h, --help            show this help message and exit
  -m {teacher,student}, --mode {teacher,student}
                        playground mode
  --env ENV             environment ID
  --algo {a2c,ddpg,dqn,ppo,her,sac,td3,qrdqn,tqc}
                        RL Algorithm
  -f FOLDER, --folder FOLDER
                        well trained teachers storage
  -p PATH_TO_STUDENT, --path-to-student PATH_TO_STUDENT
                        well trained students sotrage
  --render RENDER       render the environment(default: true)
  --testing-batch-size N
                        batch size for testing student policy (default: 1000)
```





## Policy Distillation

Distillation from trained teacher agent to pure student agent.

```bash
python policy_distillation.py --algo algo_name --env env_name 
```

> *I only tested on TD3, AntBulletEnv-v0(default) environment  so I cannot not sure that it work on other algorithms.* **PR is welcome**!

See the details below!

```bash
usage: policy_distillation.py [-h] [--env ENV] [-f FOLDER]
                              [--algo {a2c,ddpg,dqn,ppo,her,sac,td3,qrdqn,tqc}]
                              [--hidden-size HIDDEN_SIZE]
                              [--num-layers NUM_LAYERS] [--seed N]
                              [--agent-count N] [--num-teachers N]
                              [--sample-batch-size N] [--render] [--lr G]
                              [--test-interval N] [--student-batch-size N]
                              [--sample-interval N] [--testing-batch-size N]
                              [--num-student-episodes N]
                              [--loss-metric LOSS_METRIC]

Policy distillation

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment ID
  -f FOLDER, --folder FOLDER
                        Log folder
  --algo {a2c,ddpg,dqn,ppo,her,sac,td3,qrdqn,tqc}
                        RL Algorithm
  --hidden-size HIDDEN_SIZE
                        number of hidden units per layer
  --num-layers NUM_LAYERS
                        number of hidden layers
  --seed N              random seed (default: 1)
  --agent-count N       number of agents (default: 100)
  --num-teachers N      number of teacher policies (default: 1)
  --sample-batch-size N
                        expert batch size for each teacher (default: 10000)
  --render              render the environment
  --lr G                adam learnig rate (default: 1e-3)
  --test-interval N     interval between training status logs (default: 10)
  --student-batch-size N
                        per-iteration batch size for student (default: 1000)
  --sample-interval N   frequency to update expert data (default: 10)
  --testing-batch-size N
                        batch size for testing student policy (default: 10000)
  --num-student-episodes N
                        num of teacher training episodes (default: 1000)
  --loss-metric LOSS_METRIC
                        metric to build student objective
```





## Play a Distilled Agent

If you want to play a distilled_agent that we call `trained_student`,

```bash
python playground.py --mode student -p path-to-student
# For example,
# python playground.py --mode student -p '/home/user/git_storage/policy-distillation-for-control/distilled-agents/AntBulletEnv-v0_td3_1618214113.531515/student_7500_3205.61.pkl' 
# (path to ckpoint! drag & drop the file on bash terminal)
# if you changed the algorithm or environment from default, you also shold change.
```

See the details on [above](https://github.com/CUN-bjy/policy-distillation-baselines#play-a-trained-agent)!



## References

[1] 

```
@misc{rusu2016policy,
      title={Policy Distillation}, 
      author={Andrei A. Rusu and Sergio Gomez Colmenarejo and Caglar Gulcehre and Guillaume Desjardins and James Kirkpatrick and Razvan Pascanu and Volodymyr Mnih and Koray Kavukcuoglu and Raia Hadsell},
      year={2016},
      eprint={1511.06295},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[2] [Mee321/policy-distillation](https://github.com/Mee321/policy-distillation)

[3] [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) / [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) / [DLR-RM/rl-trained-agents](https://github.com/DLR-RM/rl-trained-agents)