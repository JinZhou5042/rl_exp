import socket
import os
import re
from functions import estimate_pi, my_func, combine_csv, my_func_with_threadpoolctl
import ndcctools.taskvine as vine
from tqdm import tqdm
import subprocess
import time
import glob
from resource_manager import resourceManager
import argparse
import shutil

import threading

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--port', type=int, default=9124)
    parser.add_argument('--tasks', type=int, default=10)
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--library-cores', required=True, type=int)
    parser.add_argument('--function-slots', required=True, type=int)
    args = parser.parse_args()

    size = args.size
    tasks = args.tasks
    library_cores = args.library_cores
    function_slots = args.function_slots

    q = vine.Manager(port=args.port)
    q.set_name("rl-manager")
    m = resourceManager(q)

    for i in range(0, 1):
        m.create_library('rl-library', my_func)
        m.set_cores(library_cores)
        m.set_function_slots(function_slots)
        m.set_memory(1000)
        m.install_library()

        # q.tune('wait-for-workers', 4)
        function_name = 'my_func'
        for _ in range(0, tasks):
            task = vine.FunctionCall('rl-library', function_name, size)
            m.submit(task)

        print("Waiting for results...")
        pbar = tqdm(total=tasks)
        m.start_running()
        while not q.empty():
            t = q.wait(0)
            if t:
                output = t.output
                pbar.update(1)
        pbar.close()
        m.end_running()
        print(f"tasks completed: {tasks}, time used: {round(m.time_elapsed, 3)}s")
        m.get_all_states()
        print(f"{m.all_states_df}")

        m.reset()


    combine_csv(output_csv=os.path.join('vine-run-info', 'most-recent', 'vine-logs', 'resource_consumption_report.csv'))
    
    for csv_file in glob.glob('*.csv'):
        os.remove(csv_file)

    script_path = os.getcwd()
    os.chdir('vine-run-info/most-recent/vine-logs')
    subprocess.run(['vine_graph_log', 'performance'], capture_output=True, text=True)
    os.chdir(script_path)

    msot_recent = os.readlink(os.path.join('vine-run-info', 'most-recent'))
    target_dir = os.path.join("vine-run-info", f"tasks{args.tasks}_size{args.size}_libcores{args.library_cores}_funslots{args.function_slots}_function-{function_name}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(msot_recent, target_dir)


if __name__ == '__main__':
    main()
