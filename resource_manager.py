import os
import pandas as pd
import time
import glob

class resourceManager():
    def __init__(self, manager):
        self.manager = manager
        self.library_name = None
        self.library_task = None
        self.library_cores = None
        self.function_slots = None
        self.library_memory = None

        self.time_start = None
        self.time_end = None
        self.time_elapsed = None
        self.task_count = 0

        self.all_states_df = pd.DataFrame()
        self.action_taken_timestamps = []

    def create_library(self, library_name, function):
        self.library_name = library_name
        self.library_task = self.manager.create_library_from_functions(library_name, function, add_env=True)
    
    def set_cores(self, cores):
        self.library_cores = cores
        self.library_task.set_cores(cores)

    def set_function_slots(self, function_slots):
        self.function_slots = function_slots
        self.library_task.set_function_slots(function_slots)

    def set_memory(self, memory):
        self.library_memory = memory
        self.library_task.set_memory(memory)

    def install_library(self):
        self.manager.install_library(self.library_task)

    def get_avg_task_time(self):
        if self.task_count == 0:
            print("task_count is 0")
            return 0
        if self.time_elapsed is None:
            print("time_elapsed is None")
            return 0
        return self.time_elapsed / self.task_count

    def reset(self):
        self.manager.remove_library(self.library_name)
        self.library_name = None
        self.library_task = None
        self.library_cores = None
        self.function_slots = None
        self.library_memory = None
        self.time_start = None
        self.time_end = None
        self.time_elapsed = None
        self.task_count = 0
        self.all_states_df = pd.DataFrame()

    def start_running(self):
        self.time_start = time.time()

    def submit(self, task):
        self.task_count += 1
        self.manager.submit(task)

    def end_running(self):
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start

    def get_all_states(self):
        data_dir = '.'
        # update data newly generated
        new_df = pd.DataFrame()
        for file in os.listdir(data_dir):
            if file.startswith('task') and file.endswith('.csv'):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                new_df = pd.concat([new_df, df], ignore_index=True)

        if not new_df.empty:
            new_df = new_df.sort_values(by='timestamp')
            if self.all_states_df.empty:
                self.all_states_df = new_df
            else:
                self.all_states_df = pd.concat([self.all_states_df, new_df]).drop_duplicates().sort_values(by='timestamp')

    def action_reset_function_slots(self, function_slots):
        self.get_all_states()
        self.remove_library()
        self.create_library(self.library_name, self.library_task.function)
        self.set_cores(self.library_cores)
        self.set_function_slots(function_slots)
        self.install_library()
        action_timestamp = self.all_states_df['timestamp'].iloc[-1]
        self.action_taken_timestamps.append(action_timestamp)

    def get_state_and_reward_from_last_action(self):
        self.get_all_states()
        last_action_timestamp = self.action_taken_timestamps[-1]
        states_df = self.all_states_df[self.all_states_df['timestamp'] > last_action_timestamp]
        state = {
            'avg_user_total_cpu_usage': states_df['user_total_cpu_usage'].mean(),
            'avg_user_memory_used_mb': states_df['user_memory_used_mb'].mean(),
            'avg_user_concurrent_tasks': states_df['user_concurrent_tasks'].mean(),
            'avg_bytes_sent_throughput': states_df['bytes_sent_throughput'].mean(),
            'avg_bytes_recv_throughput': states_df['bytes_recv_throughput'].mean(),
            'avg_network_latency': states_df['avg_latency'].mean(),
        }
        cpu_utilization = state["avg_user_total_cpu_usage"] / (self.library_cores * 100)
        memory_utilization = state["avg_user_memory_used_mb"] / self.library_memory
        # set a great reward for reinforcement learning agent
        reward = 1 - cpu_utilization - memory_utilization
        
        return state, reward
        return state, reward