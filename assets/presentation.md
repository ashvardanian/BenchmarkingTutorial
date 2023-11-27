---
marp: true
---

# Learning To Benchmark

By Ashot Vardanian,
Founder of [Unum.cloud](https://unum.cloud)

--

[linkedin.com/in/ashvardanian](linkedin.com/in/ashvardanian)
[t.me/ashvardanian](t.me/ashvardanian)
[ashvardanian.com](ashvardanian.com/about)

---

## The Plan for Ultimate Google Benchmark Exploration

1. Nanosecond-resolution done right 🔬
2. Bigger routines and daily programming
3. `copmare.py` the results and `perf`
4. Big data and performance addiction 💉

---

## `compare.py`: Epyc vs Threadripper

![Epyc vs Threadripper](epyc_vs_pro.png)

---

## `compare.py`: O1 vs O3

![O1 vs O3](o1_vs_o3.png)

---

## `perf`ing

```sh
sudo perf stat taskset 0xEFFFEFFFEFFFEFFFEFFFEFFFEFFFEFFF \
    ./release/main \
    --benchmark_enable_random_interleaving=true \
    --benchmark_filter=super_sort
```

---

## Results

```sh
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
  1244.984080000 seconds sy

```

---

## Performance Becomes An Addiction

### I am abusing that drug for 6 years now

* [879 GB/s Parallel Reductions in C++ & CUDA](https://unum.cloud/post/2022-01-28-reduce/).
* [Failing to Reach DDR4 Bandwidth](https://unum.cloud/post/2022-01-29-ddr4/).

Let's explore the hard cases.

---

## Try Yourself

GitHub:

* [ashvardanian/BenchmarkingTutorial](github.com/ashvardanian/BenchmarkingTutorial)
* [unum-cloud/ParallelReductions](github.com/unum-cloud/ParallelReductions)

