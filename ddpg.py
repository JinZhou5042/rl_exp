# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import ndcctools.taskvine as vine

# Lib
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import os

# Files
from noise import OrnsteinUhlenbeckActionNoise as ou_noise
from replaybuffer import Buffer
from actorcritic import Actor, Critic

plot_fig = False

# Hyperparameters
actor_lr = 0.0003
critic_lr = 0.003
minibatch_size = 64
NUM_EPISODES = 9000
num_timesteps = 300
mu = 0
sigma = 0.2
checkpoint_dir = './checkpoints/manipulator/'
buffer_size = 100000
discount = 0.9
tau = 0.001
warmup = 70
epsilon = 1.0
epsilon_decay = 1e-6

num_actions = 15
num_states = 5 + 2 + 3

id_name = 'default'

# converts observation dictionary to state tensor
# TODO: currently it's conversion between list and state tensor
def obs_to_state(state_list):
    return torch.FloatTensor(state_list).view(1, -1)

class DDPG:
    def __init__(self, manager):
        self.manager = manager

        self.start = 0
        self.end = NUM_EPISODES

    # training of the original and target actor-critic networks
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
            
        print('Training started...')
        
        action_step = 10

        all_rewards = []
        avg_rewards = []
        # for each episode 
        for episode in range(self.start, self.end):
            state = self.
