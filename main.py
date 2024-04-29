import socket
import os
import re
from functions import estimate_pi, my_func, combine_csv, my_func_with_threadpoolctl
import ndcctools.taskvine as vine
from tqdm import tqdm
import subprocess
import time
import glob
import argparse
import numpy as np
import random
from ddpg import DDPG
import shutil

import threading

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--port', type=int, default=9124)
    parser.add_argument('--tasks', type=int, default=20)
    parser.add_argument('--size', type=int, default=8000)
    parser.add_argument('--library-cores', default=16, type=int)
    parser.add_argument('--library-memory', default=1000, type=int)
    parser.add_argument('--function-slots', default=8, type=int)
    args = parser.parse_args()

    size_tasks = args.size
    num_tasks = args.tasks
    library_cores = args.library_cores
    library_memory = args.library_memory
    function_slots = args.function_slots

    # create agent
    q = vine.Manager(port=args.port)
    q.set_name("rl-manager")
    agent = DDPG(q, "rl-library", my_func, library_cores, library_memory, function_slots, size_tasks, num_tasks)

    # create tasks

    agent.train()

    exit(1)
    combine_csv(output_csv=os.path.join('vine-run-info', 'most-recent', 'vine-logs', 'resource_consumption_report.csv'))

    for csv_file in glob.glob('*.csv'):
        os.remove(csv_file)

    script_path = os.getcwd()
    os.chdir('vine-run-info/most-recent/vine-logs')
    subprocess.run(['vine_graph_log', 'performance'], capture_output=True, text=True)
    os.chdir(script_path)

    msot_recent = os.readlink(os.path.join('vine-run-info', 'most-recent'))
    target_dir = os.path.join("vine-run-info", f"num_tasks{args.num_tasks}_size{args.size}_libcores{args.library_cores}_funslots{args.function_slots}_function-{function_name}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(msot_recent, target_dir)


if __name__ == '__main__':
    main()
