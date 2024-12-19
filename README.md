Much of modern code suffers from common pitfalls: bugs, security vulnerabilities, and performance bottlenecks.
University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.
This repository provides practical examples of writing efficient C and C++ code.

![Less Slow C++](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.cpp.jpg?raw=true)

Even when an example seems over-engineered, it doesn’t make it less relevant or impractical.
The patterns discussed here often appear implicitly in large-scale software, even if most developers don’t consciously recognize them.

This is why some developers gravitate toward costly abstractions like multiple inheritance with dynamic polymorphism (e.g., `virtual` functions in C++) or using dynamic memory allocation inside loops.
They rarely design benchmarks representing real-world projects with 100K+ lines of code.
They rarely scale workloads across hundreds of cores, as required in modern cloud environments.
They rarely interface with specialized hardware accelerators that have distinct address spaces.

But we're not here to be average — we're here to be better.
We want to know the cost of unaligned memory accesses, branch prediction, CPU cache misses and the latency of different cache levels, the frequency scaling policy levels, the cost of polymorphysm and asynchronous programming, and the trade-offs between accuracy and efficiency in numerical computations.
Let's dig deeper into writing __less slow__, more efficient software.

## Quick Start

> [Jump to reading](#basics) 🔗

If you are familiar with C++ and want to go through code and measurements as you read, you can clone the repository and execute the following commands.

```sh
git clone https://github.com/ashvardanian/LessSlow.cpp.git  # Clone the repository
cd LessSlow.cpp                                             # Change the directory
cmake -B build_release -D CMAKE_BUILD_TYPE=Release          # Generate the build files
cmake --build build_release --config Release                # Build the project
build_release/less_slow                                     # Run the benchmarks
```

For brevity, the tutorial is intended for GCC and Clang compilers on Linux, but should be compatible with MacOS and Windows.
To control the output or run specific benchmarks, use the following flags:

```sh
build_release/less_slow --benchmark_format=json             # Output in JSON format
build_release/less_slow --benchmark_out=results.json        # Save the results to a file, instead of `stdout`
build_release/less_slow --benchmark_filter=std_sort         # Run only benchmarks containing `std_sort` in their name
```

> The builds will [Google Benchmark](https://github.com/google/benchmark) and [Intel's oneTBB](https://github.com/uxlfoundation/oneTBB) for the Parallel STL implementation.

To enhance stability and reproducibility, use the `--benchmark_enable_random_interleaving=true` flag which shuffles and interleaves benchmarks as described [here](https://github.com/google/benchmark/blob/main/docs/random_interleaving.md).

```sh
build_release/less_slow --benchmark_enable_random_interleaving=true
```

Google Benchmark supports [User-Requested Performance Counters](https://github.com/google/benchmark/blob/main/docs/perf_counters.md) through `libpmf`.
Note that collecting these may require `sudo` privileges.

```sh
sudo build_release/less_slow --benchmark_enable_random_interleaving=true --benchmark_format=json --benchmark_perf_counters="CYCLES,INSTRUCTIONS"
```

Alternatively, use the Linux `perf` tool for performance counter collection:

```sh
sudo perf stat taskset 0xEFFFEFFFEFFFEFFFEFFFEFFFEFFFEFFF build_release/less_slow --benchmark_enable_random_interleaving=true --benchmark_filter=super_sort
```

## Basics

### How to Benchmark and Randomness

```cpp
static void i32_addition(bm::State &state) {
    std::int32_t a = 0, b = 0, c = 0;
    for (auto _ : state)
        c = a + b;
}
```

### Parallelism and Computational Complexity

### Recursion and Branch Prediction

## Numerics

### Accuracy vs Efficiency of Standard Libraries

[![Meme IEEE 754 vs GCC](assets/meme-ieee764-vs-gnu-compiler-cover.png)](https://ashvardanian.com/posts/google-benchmark/)

### Expensive Integer Operations

### CPU Ports

### Compute vs Memory Bounds with Matrix Multiplications

### Alignment of Memory Accesses

### Non Uniform Memory Access

## Abstractions

### Virtual Functions and Polymorphism

### Ranges and Iterators

### Coroutines and Asynchronous Programming

C++20 introduces coroutines, or pre-emptive multitasking, which allows you to write asynchronous code in a synchronous manner.
Unlike a function, a coroutine can be paused and resumed at `co_await` and `co_yield` points.
In high-level languages the implementation of coroutines is universally bad and prohibitively expensive.
In C++, they are much better, but still have a cost.


## Further Reading

Many of the examples here are condensed versions of the articles on the ["Less Slow" blog](https://ashvardanian.com/tags/less-slow/).
For advanced parallel algorithm benchmarks, see [ashvardanian/ParallelReductionsBenchmark](https://github.com/ashvardanian/ParallelReductionsBenchmark).
For SIMD algorithms, check the production code at [ashvardanian/SimSIMD](https://github.com/ashvardanian/SimSIMD) and [ashvardanian/StringZilla](https://github.com/asvardanian/StringZilla), or individual articles:

- [Optimizing C++ & CUDA for High-Speed Parallel Reductions](https://ashvardanian.com/posts/cuda-parallel-reductions/)
- [Challenges in Maximizing DDR4 Bandwidth](https://ashvardanian.com/posts/ddr4-bandwidth/)
- [Comparing GCC Compiler and Manual Assembly Performance](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/)
- [Enhancing SciPy Performance with AVX-512 & SVE](https://ashvardanian.com/posts/simsimd-faster-scipy/).

