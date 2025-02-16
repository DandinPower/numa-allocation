import torch
from .allocation import zeros_numa_onnode as _zeros_numa_onnode

def zeros_numa_onnode_cpu(
    shape: tuple[int],
    dtype: torch.dtype,
    priority_numa_nodes: list[int] = None,
) -> torch.Tensor:
    """
    Allocates a NUMA-aware CPU tensor.
    
    Args:
        shape (tuple[int]): The shape of the tensor.
        dtype (torch.dtype): Data type (currently supports bfloat16, float16, float32).
        priority_numa_nodes (list[int], optional): Preferred NUMA node(s). If None, the default NUMA policy is used. Otherwise, allocation will first attempt on the leftmost node and will move to the next node only if the current node is invalid or out of free space.
    
    Returns:
        torch.Tensor: A tensor allocated according to the specified NUMA policy.
    """
    nodes = priority_numa_nodes if priority_numa_nodes is not None else []
    return _zeros_numa_onnode(shape, dtype, nodes)
