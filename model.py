import pandas as pd
import glob
import os
import torch
from torch import nn
from torch.nn import functional as F


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


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=dim_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.input_proj = nn.Linear(input_dim, dim_model)
        self.output_proj = nn.Linear(dim_model, output_dim)

    def forward(self, src):
        src = self.input_proj(src)
        src = src.unsqueeze(1)
        output = self.transformer(src, src)
        output = self.output_proj(output.squeeze(1)) 
        return F.relu(output) 
    

class RLAgentWithTransformer:
    def __init__(self, transformer_model):
        self.model = transformer_model

    def get_state(self, report_path):
        report_csv = 'resource_consumption_report.csv' 
        combine_csv(report_csv)
        state = parse_report_to_state(report_csv)
        return state
    
    def select_action(self, state):
        state_values = list(state.values())
        state_tensor = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state_tensor)
        return action.item() 
