import torch
from numa_allocation import zeros_numa_onnode_cpu
import os
import psutil
import subprocess

def get_process_name(pid=None):
    # If no PID is provided, use the current process ID.
    if pid is None:
        pid = os.getpid()
    try:
        process = psutil.Process(pid)
        return process.name()
    except psutil.Error as e:
        return f"Error retrieving process name: {e}"

def execute_command(cmd_list):
    # Run the command and capture its output as text.
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Command failed with exit code {result.returncode}"
    
def test_tensor_shape_and_zeros():
    # Create a 10x10 tensor filled with zeros.
    shape = (1024, 1024, 1024)
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=torch.float, priority_numa_nodes=[0])

    tensor_1 = zeros_numa_onnode_cpu(shape=shape, dtype=torch.float, priority_numa_nodes=[1])
    
    tensor_2 = zeros_numa_onnode_cpu(shape=shape, dtype=torch.float, priority_numa_nodes=[3])    

    
    pid = os.getpid()
    print(execute_command(["numastat", "-c", "-p" ,f"{pid}"]))

test_tensor_shape_and_zeros()