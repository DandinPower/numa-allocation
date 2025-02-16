# NUMA-Aware Tensor Allocation for PyTorch

This package provides a NUMA-aware CPU tensor allocation function for PyTorch. It allows you to allocate tensors on specific NUMA nodes or interleave them across nodes, with optional page-lock pinning for faster CUDA transfers.

## Prerequisites

Before installing and running the package, ensure that you have the following dependencies installed:

- **NUMA Libraries:**
  - `libnuma-dev`
  - `numactl` (command-line utility)
- **CUDA Toolkit:**
  - `nvcc` (for compiling the CUDA extension)
- **Python Dependencies:**
  - Python 3.12 is well test

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install the package:**

   ```bash
   pip install .
   ```

## Running Tests

A test script is provided to verify the functionality and correctness of the NUMA-aware allocation. It uses `numastat` to print the allocation status on different NUMA nodes.

**Note:** To see the expected NUMA interleaved behavior, run the tests under the NUMA control:

```bash
numactl --interleave=all python test_correctness.py
```

This command runs the tests and prints example outputs for different allocation scenarios, such as default interleaved allocation, allocation on a specific NUMA node, or fallback behavior with invalid nodes.

## API Usage Example

Below is an example of how to use the NUMA-aware allocation API in your code:

```python
import torch
from numa_allocation import zeros_numa_onnode_cpu

# Allocate a tensor with default NUMA interleaving
tensor = zeros_numa_onnode_cpu(
    shape=(1024, 1024, 1024),
    dtype=torch.float,
    pin_memory=True
)

# Allocate a tensor on a preferred NUMA node (e.g., node 0)
tensor_node0 = zeros_numa_onnode_cpu(
    shape=(1024, 1024, 1024),
    dtype=torch.float,
    pin_memory=False,
    priority_numa_nodes=[0]
)
```

In the above examples, setting `pin_memory=True` will register the memory with CUDA for faster host-to-device transfers.