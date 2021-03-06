# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import gym
from utils.models import *
from torch.optim import Adam, SGD
import torch
from torch.autograd import Variable
import random
import pickle, gzip
from utils.math import get_wasserstein, get_kl
from utils.agent_pd_baselines import AgentCollection,load_env_and_model
import numpy as np
from copy import deepcopy
from torch.nn import KLDivLoss
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

T = 0.01 #Temperature TODO: MAKE INTO ARGUMENT!

class Student(object):
    def __init__(self, args, optimizer=None):
        self.env, _ = load_env_and_model(args.env, args.algo, args.folder)

        self.num_inputs = self.env.observation_space.shape[0]*self.env.observation_space.shape[1] # fixed for atari env to do 2d convolution
        # num_actions = self.env.action_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.training_batch_size = args.student_batch_size
        self.testing_batch_size = args.testing_batch_size
        self.loss_metric = args.loss_metric
        self.policy = Policy(self.num_inputs, self.num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)
        self.agents = AgentCollection([self.env], [self.policy], 'cpu', render=args.render, num_agents=1)
        if not optimizer:
            self.optimizer = Adam(self.policy.parameters(), lr=args.lr)

    def train(self, expert_data):
        # get batch, each element is a memory object from utils/replay_memory.py
        batch = random.sample(expert_data, self.training_batch_size)

        # get the teacher's action distribution logits.
        act_logits_teacher = torch.vstack([sample[1] for sample in batch])
        # act_logits_teacher = torch.randn((self.training_batch_size, 3)) #TODO: REMOVE!!

        # get the states.
        state = torch.vstack([sample[0].reshape(self.num_inputs) for sample in batch]) / 255

        # get the student's action distribution logits by calling the student policy network form utils/models.py
        act_logits_student = self.policy.forward(state).logits

        # pytorch KL Divergece Loss.
        kl_loss = KLDivLoss(reduction='batchmean')
        loss = kl_loss(F.log_softmax(act_logits_student, dim=1), F.softmax(act_logits_teacher / T, dim=1))
        #kl_loss(F.log_softmax(act_logits_student, dim=1), F.softmax(act_logits_teacher/T, dim=1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def save(self, ckp_name):
        with gzip.open(ckp_name, 'wb') as f:
            pickle.dump(self.policy, f)

class Teacher(object):
    def __init__(self, envs, policies, args):
        self.envs = envs
        self.policies = policies
        self.expert_batch_size = args.sample_batch_size
        self.agents = AgentCollection(self.envs, self.policies, 'cpu', render=args.render, num_agents=args.agent_count)

    def get_expert_sample(self):
        return self.agents.get_expert_sample(self.expert_batch_size)


class TrainedStudent(object):
    def __init__(self, args, optimizer=None):
        self.env, _ = load_env_and_model(args.env, args.algo, args.folder)
        self.testing_batch_size = args.testing_batch_size

        self.policy = self.load(args.path_to_student)
        self.agents = AgentCollection([self.env], [self.policy], 'cpu', render=args.render, num_agents=1)

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def load(self, ckp_name):
        with gzip.open(ckp_name,'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data