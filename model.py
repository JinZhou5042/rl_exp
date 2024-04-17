import pandas as pd
import glob
import os


def combine_csv(output_csv='resource_consumption_report.csv'):
    file_pattern = "task*.csv"
    task_files = glob.glob(file_pattern)

    combined_df = pd.DataFrame()
    for file in task_files:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    if os.path.exists(output_csv):
        os.remove(output_csv)
    combined_df.to_csv(output_csv, index=False)


def parse_report_to_state(report_path):
    df = pd.read_csv(report_path)
    state = {
        'avg_cpu_usage': df['user_total_cpu_usage'].mean(),
        'avg_load1': df['load1'].mean(),
        'avg_load5': df['load5'].mean(),
        'avg_user_memory_used_gb': df['user_memory_used_gb'].mean(),
        'avg_bytes_sent_throughput': df['bytes_sent_throughput'].mean(),
        'avg_bytes_recv_throughput': df['bytes_recv_throughput'].mean(),
        'avg_latency': df['avg_latency'].mean(),
    }
    return state


class RLAgent:
    def __init__(self, model):
        self.model = model

    def get_state(self, report_path):
        report_csv = combine_csv()
        state = parse_report_to_state(report_csv)
        return state
    
    def select_action(self, state):
        action = self.model.predict(state)
        return action

