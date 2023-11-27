# C++ Benchmarking Tutorial using [Google Benchmark](https://github.com/google/benchmark)

This repository is a practical example of common pitfalls in benchmarking high-performance applications.
It's extensively-commented [source](tutorial.cxx) is also available in a more digestible [article form](https://ashvardanian.com/posts/google-benchmark/).

### Quick Start Guide

Clone the repository and execute the following commands to build and run the tutorial:

```sh
cmake -B ./build_release
cmake --build ./build_release --config Release
./build_release/tutorial

# For JSON output
./build_release/tutorial --benchmark_format=json

# For output to a file
./build_release/tutorial --benchmark_out=results.json

# To match a specific benchmark
./build_release/tutorial --benchmark_filter=i32_addition
```

### Compatibility and Special Features

While primarily designed for GNU C Compiler, this tutorial is also compatible with Clang.
Note that certain features may not work with LLVM, MSVC, ICC, NVCC, and other compilers.
It includes practical demonstrations of Parallel STL in GCC, focusing on different `std::execution` policies in the `std::sort` algorithm.
For advanced parallel algorithm benchmarks, see [ashvardanian/ParallelReductionsBenchmark](https://github.com/ashvardanian/ParallelReductionsBenchmark).

There are more articles on benchmarking in the ["Less Slow" blog](https://ashvardanian.com/tags/less-slow/):

- [Optimizing C++ & CUDA for High-Speed Parallel Reductions](https://ashvardanian.com/posts/cuda-parallel-reductions/)
- [Challenges in Maximizing DDR4 Bandwidth](https://ashvardanian.com/posts/ddr4-bandwidth/)
- [Comparing GCC Compiler and Manual Assembly Performance](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/)
- [Enhancing SciPy Performance with AVX-512 & SVE](https://ashvardanian.com/posts/simsimd-faster-scipy/).

### Advanced Google Benchmark Features

#### Random Interleaving

To enhance stability and reproducibility, use the `--benchmark_enable_random_interleaving=true` flag which shuffles and interleaves benchmarks as described [here](https://github.com/google/benchmark/blob/main/docs/random_interleaving.md).

```sh
./build_release/tutorial --benchmark_enable_random_interleaving=true
```

### Benchmark Comparison

Utilize Google Benchmark's [`compare.py` tool](https://github.com/google/benchmark/blob/main/docs/tools.md) for CLI-based comparison of benchmarking results from different JSON files.
The repository contains screenshots of the comparison of the following benchmarks:

- AMD Threadripper PRO 3995WX against Dual AMD EPYC 7302 16-Core CPUs: [screenshot](assets/benchmarks_epyc_vs_pro.png)
- AMD Threadripper PRO 3995WX with `-O3` vs `-O1` optimization levels: [screenshot](assets/benchmarks_o1_vs_o3.png)

### Performance Counters with Google Benchmark

Google Benchmark supports [User-Requested Performance Counters](https://github.com/google/benchmark/blob/main/docs/perf_counters.md) through `libpmf`.
Note that collecting these may require `sudo` privileges.

```sh
sudo ./build_release/tutorial --benchmark_enable_random_interleaving=true --benchmark_format=json --benchmark_perf_counters="CYCLES,INSTRUCTIONS"
```

Alternatively, use the Linux `perf` tool for performance counter collection:

```sh
sudo perf stat taskset 0xEFFFEFFFEFFFEFFFEFFFEFFFEFFFEFFF ./build_release/tutorial --benchmark_enable_random_interleaving=true --benchmark_filter=super_sort
```

Example output on AMD Threadripper PRO 3995WX:

```sh
 Performance counter stats for 'taskset 0xEFFFEFFFEFFFEFFFEFFFEFFFEFFFEFFF ./build_release/tutorial --benchmark_enable_random_interleaving=true --benchmark_filter=super_sort':

       23048674.55 msec task-clock                #   35.901 CPUs utilized          
           6627669      context-switches          #    0.288 K/sec                  
             75843      cpu-migrations            #    0.003 K/sec                  
         119085703      page-faults               #    0.005 M/sec                  
    91429892293048      cycles                    #    3.967 GHz                      (83.33%)
    13895432483288      stalled-cycles-frontend   #   15.20% frontend cycles idle     (83.33%)
     3277370121317      stalled-cycles-backend    #    3.58% backend cycles idle      (83.33%)
    16689799241313      instructions              #    0.18  insn per cycle         
                                                  #    0.83  stalled cycles per insn  (83.33%)
     3413731599819      branches                  #  148.110 M/sec                    (83.33%)
       11861890556      branch-misses             #    0.35% of all branches          (83.34%)

     642.008618457 seconds time elapsed

   21779.611381000 seconds user
    1244.984080000 seconds sys
```
