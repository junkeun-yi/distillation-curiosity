# original : https://github.com/Mee321/policy-distillation

import torch
import torch.autograd as autograd
import torch.nn as nn
from utils.math import *
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from collections import OrderedDict


def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=0.1, min_std=1e-6):

        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.is_disc_action = True

        # TODO: change to better network than 3 layer MLP.
        self.logits_network = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.output_size)
        )

        self.apply(_weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        logits = self.logits_network(input)

        return Categorical(logits=logits)

    def get_log_prob(self, x, actions):
        pi = self.forward(x)
        return pi.log_prob(actions)

    def select_action(self, state):
        pi = self.forward(state)
        action = pi.sample()
        return action


# class Policy(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(num_inputs, 64)
#         self.affine2 = nn.Linear(64, 64)
#
#         self.action_mean = nn.Linear(64, num_outputs)
#
#         self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
#
#         self.saved_actions = []
#         self.rewards = []
#         self.final_value = 0
#         self.is_disc_action = False
#
#     def forward(self, x, params=None):
#
#         x = torch.tanh(self.affine1(x))
#         x = torch.tanh(self.affine2(x))
#
#         action_mean = self.action_mean(x)
#         action_log_std = self.action_log_std.expand_as(action_mean)
#         action_std = torch.exp(action_log_std)
#
#         return Normal(loc=action_mean, scale=action_std)
#
#     def select_action(self, state):
#         pi = self.forward(state)
#         action = pi.sample()
#         return action
#
#     def get_log_prob(self, x, actions):
#         pi = self.forward(x)
#         return pi.log_prob(actions)

class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution