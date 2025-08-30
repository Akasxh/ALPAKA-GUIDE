# ALPAKA-GUIDE

A comprehensive guide to get started with ALPAKA (Abstraction Library for Parallel Kernel Acceleration)

## Table of Contents
- [What is ALPAKA?](#what-is-alpaka)
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Basic Example](#basic-example)
- [Key Concepts](#key-concepts)
- [Supported Backends](#supported-backends)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## What is ALPAKA?

ALPAKA is a header-only C++17 abstraction library for accelerator development. It provides performance portable parallelism by enabling the same kernel code to run efficiently on various hardware backends including:

- **CPUs** (Serial, OpenMP, TBB)
- **GPUs** (CUDA, HIP/ROCm, SYCL)
- **Other accelerators**

The library abstracts the differences between various parallel programming models, allowing developers to write code once and run it on multiple architectures without modification.

## Features

- 🚀 **Performance Portable**: Write once, run efficiently everywhere
- 🎯 **Header-only**: Easy integration, no separate compilation needed
- 🔧 **Backend Abstraction**: Support for CPU and GPU backends
- 📊 **Memory Management**: Unified memory allocation across devices
- 🔄 **Kernel Abstraction**: Single kernel code for all backends
- ⚡ **Zero-cost Abstraction**: Minimal runtime overhead

## Installation

### Prerequisites

- C++17 compatible compiler
- CMake 3.18+
- Backend-specific dependencies (CUDA SDK, ROCm, etc.)

### Using CMake

1. Clone ALPAKA:
```bash
git clone https://github.com/alpaka-group/alpaka.git
cd alpaka
```

2. Create build directory:
```bash
mkdir build && cd build
```

3. Configure with CMake:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

4. Install:
```bash
make install
```

### Using Package Managers

**vcpkg:**
```bash
vcpkg install alpaka
```

**Conan:**
```bash
conan install alpaka/[>=0.9.0]
```

## Getting Started

### 1. Include ALPAKA

```cpp
#include <alpaka/alpaka.hpp>
```

### 2. Define Your Kernel

```cpp
struct MyKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        // Your parallel code here
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        // ...
    }
};
```

### 3. Set Up Device and Execute

```cpp
int main()
{
    // Select accelerator
    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<1u>, std::size_t>;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    
    // Create queue
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    Queue queue(devAcc);
    
    // Define work division
    alpaka::Vec<alpaka::DimInt<1u>, std::size_t> const extent{1024u};
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(devAcc, extent);
    
    // Execute kernel
    alpaka::exec<Acc>(queue, workDiv, MyKernel{});
    alpaka::wait(queue);
    
    return 0;
}
```

## Basic Example

Here's a complete vector addition example:

```cpp
#include <alpaka/alpaka.hpp>
#include <vector>
#include <iostream>

struct VectorAddKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        float const* const a,
        float const* const b,
        float* const c,
        std::size_t const numElements) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        
        if(globalThreadIdx < numElements)
        {
            c[globalThreadIdx] = a[globalThreadIdx] + b[globalThreadIdx];
        }
    }
};

int main()
{
    // Define accelerator
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    
    // Get device
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::getDevByIdx<alpaka::DevCpu>(0u);
    
    // Create queue
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    Queue queue(devAcc);
    
    // Problem size
    constexpr std::size_t numElements = 1024u;
    
    // Allocate host memory
    auto hostA = alpaka::allocBuf<float, Idx>(devHost, numElements);
    auto hostB = alpaka::allocBuf<float, Idx>(devHost, numElements);
    auto hostC = alpaka::allocBuf<float, Idx>(devHost, numElements);
    
    // Initialize data
    for(std::size_t i = 0u; i < numElements; ++i)
    {
        alpaka::getPtrNative(hostA)[i] = static_cast<float>(i);
        alpaka::getPtrNative(hostB)[i] = static_cast<float>(i) * 2.0f;
    }
    
    // Allocate device memory
    auto devA = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    auto devB = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    auto devC = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    
    // Copy data to device
    alpaka::memcpy(queue, devA, hostA);
    alpaka::memcpy(queue, devB, hostB);
    
    // Set up work division
    alpaka::Vec<Dim, Idx> const extent{numElements};
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(devAcc, extent);
    
    // Execute kernel
    alpaka::exec<Acc>(
        queue,
        workDiv,
        VectorAddKernel{},
        alpaka::getPtrNative(devA),
        alpaka::getPtrNative(devB),
        alpaka::getPtrNative(devC),
        numElements);
    
    // Copy result back
    alpaka::memcpy(queue, hostC, devC);
    alpaka::wait(queue);
    
    // Verify result
    bool success = true;
    for(std::size_t i = 0u; i < numElements; ++i)
    {
        float expected = static_cast<float>(i) * 3.0f;
        if(std::abs(alpaka::getPtrNative(hostC)[i] - expected) > 1e-5f)
        {
            success = false;
            break;
        }
    }
    
    std::cout << "Vector addition " << (success ? "succeeded" : "failed") << std::endl;
    return success ? 0 : 1;
}
```

## Key Concepts

### Accelerators
- **AccCpuSerial**: Single-threaded CPU execution
- **AccCpuOmp2Blocks**: OpenMP parallel blocks
- **AccCpuOmp2Threads**: OpenMP parallel threads
- **AccCpuTbbBlocks**: Intel TBB parallel blocks
- **AccGpuCudaRt**: NVIDIA CUDA
- **AccGpuHipRt**: AMD HIP/ROCm

### Work Division
ALPAKA uses a hierarchical work division model:
- **Grids**: Contain blocks
- **Blocks**: Contain threads
- **Elements**: Processed by threads

### Memory Management
- Unified buffer allocation: `alpaka::allocBuf<T, Idx>(device, extent)`
- Memory copying: `alpaka::memcpy(queue, dst, src)`
- Host and device memory abstraction

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| CPU Serial | Single-threaded CPU | None |
| CPU OpenMP | Multi-threaded CPU with OpenMP | OpenMP |
| CPU TBB | Multi-threaded CPU with Intel TBB | Intel TBB |
| CUDA | NVIDIA GPU | CUDA SDK |
| HIP | AMD GPU | ROCm/HIP |
| SYCL | Cross-platform parallel computing | SYCL implementation |

## Documentation

- 📚 **Official Documentation**: [alpaka.readthedocs.io](https://alpaka.readthedocs.io/)
- 🔗 **GitHub Repository**: [github.com/alpaka-group/alpaka](https://github.com/alpaka-group/alpaka)
- 📖 **API Reference**: [alpaka-group.github.io/alpaka](https://alpaka-group.github.io/alpaka/)
- 🎓 **Tutorials**: [alpaka.readthedocs.io/en/latest/usage/tutorial.html](https://alpaka.readthedocs.io/en/latest/usage/tutorial.html)

## Examples

For more examples, check out:
- [ALPAKA Examples Repository](https://github.com/alpaka-group/alpaka/tree/develop/example)
- Matrix multiplication
- Reduction operations
- Memory patterns
- Multi-device programming

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Guidelines
- Follow the existing code style
- Document new features
- Include examples for new functionality
- Test on multiple backends when possible

## License

This guide is released under the MIT License. ALPAKA itself is licensed under the MPL-2.0 License.

---

**Happy parallel programming with ALPAKA! 🚀**

For questions, issues, or discussions, please visit the [ALPAKA GitHub repository](https://github.com/alpaka-group/alpaka) or join the community discussions.