# C++ Benchmarking Tutorial

This repository is a practical example of common pitfalls in benchmarking high-performance applications.
If you are interested in more advanced benchmarks - check out the [unum-cloud/ParallelReductions](https://github.com/unum-cloud/ParallelReductions) and the two following articles:

* [879 GB/s Parallel Reductions in C++ & CUDA](https://unum.cloud/post/2022-01-28-reduce/).
* [Failing to Reach DDR4 Bandwidth](https://unum.cloud/post/2022-01-29-ddr4/).

Run it with a single-line command:

```sh
mkdir -p release && cd release && cmake .. && make && ./main ; cd ..
```

Dependencies will be fetched, but it's expected that you have a modern GCC compiler.
Some parts of the tutorial will not work on LLVM, MSVC, ICC, NVCC and other compilers.

## Lesser known GBench features

* [Random Interleaving](https://github.com/google/benchmark/blob/main/docs/random_interleaving.md) with `--benchmark_enable_random_interleaving=true`.
* [User-Requested Performance Counters](https://github.com/google/benchmark/blob/main/docs/perf_counters.md) via [`libpmf`](http://perfmon2.sourceforge.net/).
* [Comparing with previous results](https://github.com/google/benchmark/blob/main/docs/tools.md) with `compare.py`.

So running command changes to:

```sh
./release/main --benchmark_enable_random_interleaving=true --benchmark_format=json --benchmark_perf_counters="CYCLES,INSTRUCTIONS"
```

## Let's compare the results with O1, O2 and O3

