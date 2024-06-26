# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import ndcctools.taskvine as vine

# Lib
import json
import numpy as np
import copy
import random
import glob
from copy import deepcopy
import matplotlib.pyplot as plt
# from IPython import display
import os
import pandas as pd
import time

# Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replay_buffer import Buffer
from models import Actor, Critic

# Hyperparameters
ACTOR_LR = 0.0003
CRITIC_LR = 0.001
MINIBATCH_SIZE = 8
NUM_EPISODES = 9000
NUM_TIMESTEPS = 300
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './checkpoints/manipulator/'
BUFFER_SIZE = 100000
DISCOUNT = 0.9
TAU = 0.001
REPLAY_BUFFER_THRESHOLD = 8
EPSILON = 1.0
EPSILON_DECAY = 1e-6

NUM_STATES = 10                 # 
NUM_ACTIONS = 9                 # (0, 1, 2) for library_cores, (3, 4, 5) for library_memory, (6, 7, 8) for function_slots
                                # 0 for no change, 1 for increase, 2 for decrease

# converts observation dictionary to state tensor
# TODO: currently it's conversion between list and state tensor
def obs_to_state(state_list):
    return torch.FloatTensor(state_list).view(1, -1)

class DDPG:
    def __init__(self, manager, library_name, function, library_cores, library_memory, function_slots, size_tasks, num_tasks):
        for csv_file in glob.glob('*.csv'):
            os.remove(csv_file)

        self.manager = manager
        self.library_name = library_name
        self.library_cores = library_cores
        self.function_slots = function_slots
        self.library_memory = library_memory
        self.function = function
        self.library_task = None

        self.create_library('rl-library', self.function, self.library_cores, self.library_memory, self.function_slots)
        
        self.size_tasks = size_tasks
        self.num_tasks = num_tasks

        self.MAX_LIBRARY_CORES = 16
        self.MAX_LIBRARY_MEMORY = 1000000
        self.MAX_FUNCTION_SLOTS = 16
        
        self.all_function_calls = []
        self.submitted_function_calls = set()  # record the submitted tasks

        self.all_states_df = pd.DataFrame()
        self.start_timestamps = []
        self.batch_end_timestamps = []
    
        self.state_dim = NUM_STATES
        self.action_dim = NUM_ACTIONS
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.noise = OUNoise(mu=np.zeros(self.action_dim), sigma=SIGMA)
        self.mseloss = nn.MSELoss()
        self.replay_buffer = Buffer(BUFFER_SIZE)

        self.functions_per_step = 20
        self.batch_size = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.replay_buffer_threshold = REPLAY_BUFFER_THRESHOLD
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.tau = TAU
        self.current_state = None

        self.reward_graph = []

        self.num_epochs = NUM_EPISODES

        self.available_actions = {
            'library_cores': [0, -1, 1],
            'library_memory': [0, -10, 10],
            'function_slots': [0, -1, 1],
        }
        
        self.action_ranges = {
            'library_cores': (0, 3),
            'library_memory': (3, 6),
            'function_slots': (6, 9),
        }

    def create_library(self, library_name, function, library_cores, library_memory, function_slots):
        """only accept one function for now"""
        self.library_name = library_name
        self.function = function
        self.library_cores = library_cores
        self.library_memory = library_memory
        self.function_slots = function_slots

        self.library_task = self.manager.create_library_from_functions(library_name, function, add_env=False)
        self.library_task.set_env_var('PATH', '/scratch365/jzhou24/env/bin/:$PATH')

        self.library_task.set_cores(library_cores)
        self.library_task.set_memory(library_memory)
        self.library_task.set_function_slots(function_slots)

        self.manager.install_library(self.library_task)

    def remove_library(self):
        self.manager.remove_library(self.library_name)
        self.library_name = None
        self.library_task = None
        self.library_cores = None
        self.function_slots = None
        self.library_memory = None
        time.sleep(5)

    def syn_states(self):
        data_dir = '.'
        data_frames = [] 

        for file in os.listdir(data_dir):
            if file.startswith('task') and file.endswith('.csv'):
                file_path = os.path.join(data_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    data_frames.append(df)
                    os.remove(file_path) 
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        if data_frames:
            new_df = pd.concat(data_frames, ignore_index=True)
            if 'timestamp' in new_df.columns:
                new_df.sort_values(by='timestamp', inplace=True)
                if self.all_states_df.empty:
                    self.all_states_df = new_df
                else:
                    self.all_states_df = pd.concat([self.all_states_df, new_df], ignore_index=True)
        
        if not self.all_states_df.empty:
            self.all_states_df.sort_values(by='timestamp', inplace=True)
            self.all_states_df.to_csv('all_states.csv', index=False) 


    def reinstall_library(self, library_cores, library_memory, function_slots):
        self.manager.remove_library(self.library_name)
        time.sleep(5)
        self.create_library('rl-library', self.function, library_cores, library_memory, function_slots)

    def get_action(self, action, action_range_start, action_range_end, available_actions):
        if action_range_start <= action < action_range_end:
            return available_actions[action - action_range_start]
        return 0

    def get_state(self):
        self.syn_states()
        # if there is no state data, return the initial state
        if not self.start_timestamps:
            state = {
                'avg_user_total_cpu_usage': 0,
                'avg_user_memory_used_mb': 0,
                'avg_user_concurrent_tasks': 0,
                'avg_bytes_sent_throughput': 0,
                'avg_bytes_recv_throughput': 0,
                'avg_network_latency': 0,
                'library_cores': self.library_cores,
                'library_memory': self.library_memory,
                'function_slots': self.function_slots,
                'remaining_function_calls': len(self.all_function_calls),
            }
            return state
        # get the last batch of states
        start_timestamp = self.start_timestamps[-1]
        states_df = self.all_states_df[self.all_states_df['timestamp'] > start_timestamp]
        state = {
            'avg_user_total_cpu_usage': states_df['user_total_cpu_usage'].mean(),
            'avg_user_memory_used_mb': states_df['user_memory_used_mb'].mean(),
            'avg_user_concurrent_tasks': states_df['user_concurrent_tasks'].mean(),
            'avg_bytes_sent_throughput': states_df['bytes_sent_throughput'].mean(),
            'avg_bytes_recv_throughput': states_df['bytes_recv_throughput'].mean(),
            'avg_network_latency': states_df['avg_latency'].mean(),
            'library_cores': self.library_cores,
            'library_memory': self.library_memory,
            'function_slots': self.function_slots,
            'remaining_function_calls': len(self.all_function_calls) - len(self.submitted_function_calls),
        }
        return state
    
    def get_reward(self):
        state = self.get_state()
        print(f"state = {state}")
        cpu_util = state["avg_user_total_cpu_usage"] / (self.library_cores * 100)
        mem_util = state["avg_user_memory_used_mb"] / self.library_memory
        if cpu_util < 0 or mem_util < 0:
            print(f"error: cpu_util = {cpu_util}, mem_util = {mem_util}")
            exit(1)
        if cpu_util > 1.2 or mem_util > 1.2:
            reward = -abs(cpu_util + mem_util - 2)
        elif cpu_util + mem_util <= 2.4:
            reward = cpu_util + mem_util
        else:
            reward = -abs(cpu_util + mem_util - 2)
        return reward

    def get_q_target(self, next_state_batch, reward_batch, terminal_batch):
        target_batch = torch.FloatTensor(reward_batch)
        non_final_mask = 1 - torch.ByteTensor(terminal_batch)
        next_state_batch = torch.cat(next_state_batch)
        next_action_batch = self.target_actor(next_state_batch)
        q_next = self.target_critic(next_state_batch, next_action_batch)
        
        # if the next state is terminal, the target is the reward
        non_final_mask = self.discount * non_final_mask.float()
        target_batch += non_final_mask * q_next.squeeze().data
        
        return target_batch

    # weighted average update of the target network and original network
    # Inputs: target actor(critic) and original actor(critic)
    def update_targets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

    # Inputs: Current state of the episode
    # Output: the action which maximizes the Q-value of the current state-action pair
    def get_max_action(self, current_state):
        noise = self.epsilon * torch.FloatTensor(self.noise())
        action = self.actor(current_state) + noise
        
        # get the max
        action_list = action.tolist()[0]
        max_action = max(action_list)
        max_index = action_list.index(max_action)

        return max_index, action

    def register_function_calls(self, function_calls):
        self.all_function_calls = function_calls

    def submit_function_calls(self, num):
        tasks_submitted = 0
        for funcall in self.all_function_calls:
            if funcall not in self.submitted_function_calls:
                funcall.set_env_var('PATH', '/scratch365/jzhou24/env/bin/:$PATH')
                self.manager.submit(funcall)
                self.submitted_function_calls.add(funcall)
                tasks_submitted += 1
                if tasks_submitted >= num:
                    break
        return tasks_submitted

    def get_batch_time_elapsed(self):
        if not self.start_timestamps or not self.batch_end_timestamps:
            print("start_timestamps or batch_end_timestamps is empty")
            exit(1)
        time_start, time_end = self.start_timestamps[-1], self.batch_end_timestamps[-1]

        if time_start > time_end:
            print(f"time_start = {time_start}, time_end = {time_end}")
            exit(1)
        return time_end - time_start

    def run_batch(self):
        from tqdm import tqdm
        self.start_timestamps.append(time.time())
        tasks_submitted = 0
        for funcall in self.all_function_calls:
            if funcall not in self.submitted_function_calls:
                funcall.set_env_var('PATH', '/scratch365/jzhou24/env/bin/:$PATH')
                self.manager.submit(funcall)
                self.submitted_function_calls.add(funcall)
                tasks_submitted += 1
                if tasks_submitted >= self.functions_per_step:
                    break
        pbar = tqdm(total=tasks_submitted)
        while not self.manager.empty():
            t = self.manager.wait(0)
            if t:
                output = t.output
                pbar.update(1)
        self.batch_end_timestamps.append(time.time())
        pbar.close()

    def epoch_reset(self):
        # self.all_states_df = pd.DataFrame()
        self.start_timestamps = []
        self.batch_end_timestamps = []
        self.create_all_function_calls()
        self.submitted_function_calls = set()
        # randomize the library cores, library memory, function slots
        self.library_cores = random.randint(1, 16)
        self.library_memory = random.randint(100, 1000)
        self.function_slots = random.randint(2, 16)
        self.current_state = {
            'avg_user_total_cpu_usage': 0,
            'avg_user_memory_used_mb': 0,
            'avg_user_concurrent_tasks': 0,
            'avg_bytes_sent_throughput': 0,
            'avg_bytes_recv_throughput': 0,
            'avg_network_latency': 0,
            'library_cores': self.library_cores,
            'library_memory': self.library_memory,
            'function_slots': self.function_slots,
            'remaining_function_calls': len(self.all_function_calls),
        }

    def create_all_function_calls(self):
        self.all_function_calls = []
        for _ in range(0, self.num_tasks):
            # give a random size
            rand_size = int(np.random.uniform(self.size_tasks, 100))
            # make sure the size > 100 and < 30000
            rand_size = max(100, min(2000, rand_size))
            self.all_function_calls.append(vine.FunctionCall('rl-library', 'my_func', self.size_tasks))

    # training of the original and target actor-critic networks
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        print('Training started...')

        # for each episode
        for epoch in range(0, self.num_epochs):
            epoch_reward = 0
            self.epoch_reset()
            num_steps = len(self.all_function_calls) // self.functions_per_step

            for step in range(num_steps):
                current_state = self.current_state
                
                # get maximizing action
                current_state_tensor = obs_to_state(list(current_state.values()))
                self.actor.eval()

                action_index, action = self.get_max_action(current_state_tensor)

                # get the action
                library_cores_action = self.get_action(action_index, *self.action_ranges['library_cores'], self.available_actions['library_cores'])
                library_memory_action = self.get_action(action_index, *self.action_ranges['library_memory'], self.available_actions['library_memory'])
                function_slots_action = self.get_action(action_index, *self.action_ranges['function_slots'], self.available_actions['function_slots'])
                
                self.actor.train()
                
                # take the action
                self.library_cores += library_cores_action
                self.library_memory += library_memory_action
                self.function_slots += function_slots_action
                self.library_cores = min(self.library_cores, self.MAX_LIBRARY_CORES)
                self.library_memory = min(self.library_memory, self.MAX_LIBRARY_MEMORY)
                self.function_slots = max(2, min(self.function_slots, self.MAX_FUNCTION_SLOTS))

                self.reinstall_library(self.library_cores, self.library_memory, self.function_slots)
                self.run_batch()

                next_state = self.get_state()

                self.current_state = next_state
                reward = self.get_reward()
                epoch_reward += reward

                print(f"epoch = {epoch}, step = {step}, action = ({self.library_cores}, {self.library_memory}, {self.function_slots}), reward = {reward}")

                next_state_tensor = obs_to_state(list(next_state.values()))
                # if no available tasks, the terminal_status is True
                terminal_status = len(self.all_function_calls) == len(self.submitted_function_calls)
                self.replay_buffer.append((current_state_tensor, action, next_state_tensor, reward, terminal_status))

                # if the replay buffer is full, start training
                if len(self.replay_buffer) >= self.replay_buffer_threshold:
                    print("start training...")
                    sample_batch = self.replay_buffer.sample_batch(self.batch_size)
                    if sample_batch:
                        current_state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = sample_batch
                        current_state_batch = torch.cat(current_state_batch)
                        action_batch = torch.cat(action_batch)
                        print(f"current_state_batch = {current_state_batch}, action_batch = {action_batch}, next_state_batch = {next_state_batch}, reward_batch = {reward_batch}, terminal_batch = {terminal_batch}")
                        predicted_q_values = self.critic(current_state_batch, action_batch)
                        target_q_values = self.get_q_target(next_state_batch, reward_batch, terminal_batch)
                        
                        print(f"predicted_q_values = {predicted_q_values}, target_q_values = {target_q_values}")
                        # critic update
                        self.critic_optimizer.zero_grad()
                        critic_loss = self.mseloss(predicted_q_values.squeeze(), target_q_values)
                        print(f"critic_loss = {critic_loss}")
                        critic_loss.backward()
                        self.critic_optimizer.step()

                        # actor update
                        self.actor_optimizer.zero_grad()
                        actor_loss = -torch.mean(self.critic(current_state_batch, self.actor(current_state_batch)))
                        print(f"actor_loss = {actor_loss}")
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # update target networks
                        self.update_targets(self.target_actor, self.actor)
                        self.update_targets(self.target_critic, self.critic)
                        self.epsilon -= self.epsilon_decay
                    else:
                        print("Not enough samples to train.")
                
            print(f"epoch_reward = {epoch_reward}")
            self.reward_graph.append(epoch_reward)
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        print(f'saving checkpoint to {self.checkpoint_dir}...')
        checkpoint_name = self.checkpoint_dir + 'epoch{}.pth.tar'.format(epoch)
        checkpoint = {
            'epoch': epoch,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
            'reward_graph': self.reward_graph,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, checkpoint_name)

    def locad_checkpoint(self, checkpoint_name):
        if not os.path.exists(checkpoint_name):
            print(f"checkpoint {checkpoint_name} does not exist")
            return
        print(f"loading checkpoint {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.replay_buffer = checkpoint['replay_buffer']
        self.reward_graph = checkpoint['reward_graph']
        self.epsilon = checkpoint['epsilon']

    def __del__(self):
        # remove the csv files, should start with 'task'
        for csv_file in glob.glob('*.csv'):
            if csv_file.startswith('task'):
                os.remove(csv_file)