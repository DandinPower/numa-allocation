#include <torch/extension.h>
#include <numa.h>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define PAGE_SIZE 4096

torch::Tensor zeros_numa_onnode(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    std::vector<int64_t> priority_nodes) {

  if (dtype != at::kBFloat16 && dtype != at::kHalf && dtype != at::kFloat) {
    throw std::runtime_error("Unsupported dtype. Only bfloat16, float16, and float32 are supported.");
  }

  size_t numel = 1;
  for (auto s : sizes) {
    numel *= s;
  }
  size_t element_size = 0;
  if (dtype == at::kBFloat16 || dtype == at::kHalf)
    element_size = 2;
  else if (dtype == at::kFloat)
    element_size = 4;
  size_t bytes = numel * element_size;
  size_t alloc_size = ((bytes + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;


  void* ptr = nullptr;
  bool allocated_on_numa = false;
  // If a preferred node list is provided, try each node in order.
  if (!priority_nodes.empty()) {
    int max_nodes = numa_max_node(); // Highest node id available.
    for (auto node : priority_nodes) {
      if (node < 0 || node >= max_nodes) continue; // Skip invalid nodes.
      ptr = numa_alloc_onnode(alloc_size, node);
      if (ptr != nullptr) {
        allocated_on_numa = true;
        break;
      }
    }
  }
  // Fallback: use default aligned allocation if NUMA allocation failed or no nodes provided.
  if (ptr == nullptr) {
    int ret = posix_memalign(&ptr, PAGE_SIZE, alloc_size);
    if (ret != 0 || ptr == nullptr) {
      throw std::bad_alloc();
    }
  }
  std::memset(ptr, 0, alloc_size);

  // Define a custom deleter to free the memory appropriately.
  auto deleter = [allocated_on_numa, alloc_size](void* p) {
    if (allocated_on_numa) {
      numa_free(p, alloc_size);
    } else {
      free(p);
    }
  };

  // Create tensor using the allocated memory.
  auto options = torch::TensorOptions().dtype(dtype).device(at::kCPU);
  auto tensor = torch::from_blob(ptr, sizes, deleter, options);
  return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zeros_numa_onnode", &zeros_numa_onnode, "NUMA aware zero initialization");
}