import torch
import os
import subprocess
from numa_allocation import zeros_numa_onnode_cpu

def get_numastat_output() -> None:
    """Show the numastat command for the current process."""
    
    def run_command(cmd: list[str]) -> str:
        """Run a command and return its output."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Command failed with exit code {result.returncode}"
        return result.stdout.strip()
    
    pid = os.getpid()
    output = run_command(["numastat", "-c", "-p", str(pid)])
    return output

def test_pin_memory():
    dtype = torch.float
    shape = (1024, 1024, 1024)
    pin_memory = False
    
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory)
    assert tensor.is_pinned() == False
    
    pin_memory = True
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory)
    assert tensor.is_pinned() == True

    print("Passed test_pin_memory!")

def test_tensor_metadata():
    dtypes = [torch.bfloat16, torch.float16, torch.float]
    shape = (1024, 1024, 1024)
    pin_memory = False
    
    for dtype in dtypes:
        tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory)
        
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        assert torch.all(tensor == 0), "Tensor is not zero initialized"

    print("Passed test_tensor_metadata!")
    

def test_priority_numa_nodes_default():
    shape = (1024, 1024, 1024)
    dtype = torch.float
    pin_memory = False
    
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory)
    print(get_numastat_output())
    del tensor
    """
    You should see the memory allocation is interleaved around all numa nodes since when we run under numactl --interleave=all, the memory allocation will round robin on different nodes
        Per-node process memory usage (in MBs) for PID 3509020 (python)
            Node 0 Node 1 Node 2 Node 3 Node 4 Total
            ------ ------ ------ ------ ------ -----
    Huge          0      0      0      0      0     0
    Heap         17     17     17     17     17    86
    Stack         0      0      0      0      0     0
    Private     997    846    847    843    843  4376
    -------  ------ ------ ------ ------ ------ -----
    Total      1014    864    864    860    860  4462
    """
    
def test_priority_numa_nodes_with_one_value():
    shape = (1024, 1024, 1024)
    dtype = torch.float
    pin_memory = False

    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[0])
    print(get_numastat_output())
    del tensor
    """
    You should see the memory allocation is only on node we specific.
    Example:
    Per-node process memory usage (in MBs) for PID 3508911 (python)
            Node 0 Node 1 Node 2 Node 3 Node 4 Total
            ------ ------ ------ ------ ------ -----
    Huge          0      0      0      0      0     0
    Heap         17     17     17     17     17    86
    Stack         0      0      0      0      0     0
    Private    4274     27     27     24     24  4376
    -------  ------ ------ ------ ------ ------ -----
    Total      4291     44     45     41     41  4462
    """
    
def test_priority_numa_nodes_with_multiple_value():
    shape = (1024, 1024, 1024)
    dtype = torch.float
    pin_memory = False

    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[0, 1])
    print(get_numastat_output())
    del tensor
    """
    You should see the memory allocation is only on the leftmost node we specific. because it only will use another nodes when the leftmost node is invalid.
    Example:
    Per-node process memory usage (in MBs) for PID 3508798 (python)
            Node 0 Node 1 Node 2 Node 3 Node 4 Total
            ------ ------ ------ ------ ------ -----
    Huge          0      0      0      0      0     0
    Heap         17     17     17     17     17    86
    Stack         0      0      0      0      0     0
    Private    4274     27     28     24     24  4376
    -------  ------ ------ ------ ------ ------ -----
    Total      4291     44     45     41     41  4462
    """

def test_priority_numa_nodes_with_invalid_nodes():
    shape = (1024, 1024, 1024)
    dtype = torch.float
    pin_memory = False

    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[-1])
    print(get_numastat_output())
    del tensor
    """
    You should see the memory allocation is interleaved around all numa nodes, because the node is invalid so it will turn back to default policy.
    Example:
    Per-node process memory usage (in MBs) for PID 3508658 (python)
            Node 0 Node 1 Node 2 Node 3 Node 4 Total
            ------ ------ ------ ------ ------ -----
    Huge          0      0      0      0      0     0
    Heap         17     17     17     17     17    86
    Stack         0      0      0      0      0     0
    Private     997    846    847    843    843  4376
    -------  ------ ------ ------ ------ ------ -----
    Total      1014    864    864    860    860  4462
    """
    
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[-1, 1])
    print(get_numastat_output())
    del tensor
    """
    You should see the memory allocation is on the second node, because the first node is invalid, so it will allocate on next node.
    
    Example: 
    Per-node process memory usage (in MBs) for PID 3508658 (python)
            Node 0 Node 1 Node 2 Node 3 Node 4 Total
            ------ ------ ------ ------ ------ -----
    Huge          0      0      0      0      0     0
    Heap         17     17     17     17     17    86
    Stack         0      0      0      0      0     0
    Private     178   4123     28     24     24  4376
    -------  ------ ------ ------ ------ ------ -----
    Total       195   4140     45     41     41  446
    """

test_pin_memory()
test_tensor_metadata()
test_priority_numa_nodes_default()
test_priority_numa_nodes_with_one_value()
test_priority_numa_nodes_with_multiple_value()
test_priority_numa_nodes_with_invalid_nodes()