import socket
import os
import re
from functions import estimate_pi, my_func, combine_csv, my_func_with_threadpoolctl
import ndcctools.taskvine as vine
from tqdm import tqdm
import subprocess
import time
import argparse
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

    for i in range(0, 3):
        libtask = q.create_library_from_functions('rl-library', my_func, add_env=True)
        function_slots -= 2
        libtask.set_cores(library_cores)
        libtask.set_function_slots(function_slots)
        q.install_library(libtask)

        # q.tune('wait-for-workers', 4)
        function_name = 'my_func'
        for _ in range(0, tasks):
            task = vine.FunctionCall('rl-library', function_name, size)
            q.submit(task)

        print("Waiting for results...")
        pbar = tqdm(total=tasks)
        time_start = time.time()
        while not q.empty():
            t = q.wait(0)
            if t:
                output = t.output
                # print(output)
                pbar.update(1)
        pbar.close()
        q.remove_library('rl-library')
        print(f"tasks completed: {tasks}, time used: {round(time.time() - time_start, 4)}s")

    combine_csv(output_csv=os.path.join('vine-run-info', 'most-recent', 'vine-logs', 'resource_consumption_report.csv'))
    subprocess.run('rm *.csv && cd vine-run-info/most-recent/vine-logs &&  vine_graph_log performance', shell=True, capture_output=True, text=True)
    
    msot_recent = os.readlink(os.path.join('vine-run-info', 'most-recent'))
    target_dir = os.path.join("vine-run-info", f"tasks{args.tasks}_size{args.size}_libcores{args.library_cores}_funslots{args.function_slots}_function-{function_name}")
    subprocess.run(f"mv {msot_recent} {target_dir}", shell=True, capture_output=True, text=True)



if __name__ == '__main__':
    main()
