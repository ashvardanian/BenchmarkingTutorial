/**
 *  @brief  Low-level microbenchmarks designed for educational purposes.
 *  @file   less_slow.cpp
 *  @author Ash Vardanian
 */
#include <algorithm> // `std::sort`
#include <cassert>   // `assert`
#include <cmath>     // `std::pow`
#include <cstdint>   // `std::int32_t`
#include <cstring>   // `std::memcpy`, `std::strcmp`
#include <execution> // `std::execution::par_unseq`
#include <fstream>   // `std::ifstream`
#include <iterator>  // `std::random_access_iterator_tag`
#include <memory>    // `std::assume_aligned`
#include <new>       // `std::launder`
#include <random>    // `std::mt19937`
#include <vector>    // `std::algorithm`

#include <benchmark/benchmark.h>

namespace bm = benchmark;

#pragma region - Basics

#pragma region How to Benchmark and Randomness

/**
 *  Using Google Benchmark is simple. You define a C++ function, and then you register it
 *  with provided C macros. The suite will invoke your function, passing a `State` object,
 *  that dynamically chooses the number of loops to run based on the time it takes to execute
 *  each cycle.
 *
 *  For simplicity, let's start by benchmarking the most basic operation - 32-bit integer addition,
 *  universally natively supported by every modern CPU, be it x86, ARM, or RISC-V, 32-bit or 64-bit,
 *  big-endian or little-endian.
 */

static void i32_addition(bm::State &state) {
    std::int32_t a = 0, b = 0, c = 0;
    for (auto _ : state)
        c = a + b;
    (void)c; // Silence "variable `c` set but not used" warning
}

BENCHMARK(i32_addition);

/**
 *  Trivial kernels operating on constant values are not the easiest candidates for benchmarking.
 *  The compiler can easily optimize them out, and the CPU can predict the result... showing "0 ns",
 *  zero nanoseconds per iteration.
 *
 *  Unfortunately, no operation runs this fast on the computer. On a 3 GHz CPU, you would perform
 *  3 Billion ops every second. So, each would take 0.33 ns, not 0 ns. If we change the compilation
 *  settings discarding the @b `-O3` flag for "Release build" optimizations, we may see a non-zero value,
 *  but it won't be representative of the real-world performance.
 *
 *  Another thing we can try - is generating random inputs on the fly with @b `std::rand()`, one of the
 *  most controversial operations in the C standard library.
 */

static void i32_addition_random(bm::State &state) {
    std::int32_t c = 0;
    for (auto _ : state)
        c = std::rand() + std::rand();
    (void)c; // Silence "variable `c` set but not used" warning
}

BENCHMARK(i32_addition_random);

/**
 *  Running this will report @b 25ns, or about 100 CPU cycles. Is integer addition really that expensive?
 *  It's used all the time, even when you are accessing @b `std::vector` elements and need to compute the
 *  memory address from the pointer and the index passed to the @b `operator[]` or `at()` functions.
 *
 *  The answer is - no, it's not. The addition itself takes a single CPU cycle, and it's very fast.
 *  Chances are we just benchmarked something else... the `std::rand()` function.
 *
 *  What if we could ask Google Benchmark to simply ignore the time spent in the `std::rand()` function?
 *  There are `PauseTiming` and `ResumeTiming` functions just for that!
 */

static void i32_addition_paused(bm::State &state) {
    std::int32_t a = 0, b = 0, c = 0;
    for (auto _ : state) {
        state.PauseTiming();
        a = std::rand(), b = std::rand();
        state.ResumeTiming();
        bm::DoNotOptimize(c = a + b);
    }
}

BENCHMARK(i32_addition_paused);

/**
 *  Those `PauseTiming` and `ResumeTiming` functions, however, are not free either.
 *  In current implementation, they can easily take @b ~127 ns, or around 300 CPU cycles.
 *  Clearly useless in our case, but there is a good reusable recipe!
 *
 *  A typical pattern when implementing a benchmark is to initialize with a random value, and then
 *  define a very cheap update policy that won't affect the latency much but will update the inputs.
 *  Increments, bit shifts, and bit rotations are a common choice! It's also a good idea to use
 *  native @b CRC32 and @b AES instructions to produce random state, as its often done in StringZilla.
 *  Another common approach is to use integer multiplication, often derived from the Golden Ratio,
 *  as in Knuth multiplicative hash function (with `2654435761`).
 *
 *  @see StringZilla: https://github.com/ashvardanian/stringzilla
 */

static void i32_addition_randomly_initialized(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) + (++b));
}

BENCHMARK(i32_addition_randomly_initialized);

/**
 *  On x86 the `i32_addition_randomly_initialized` performs two @b `inc` instructions and an @b `add` instruction.
 *  This should take less than @b 0.7ns on a modern CPU. The first cycle was spent incrementing `a' and `b`
 *  on different Arithmetic Logic Units (ALUs) of the same core, while the second performed the
 *  final accumulation. So at least @b 97% of the benchmark was just spent in the `std::rand()` function...
 *  even in a single-threaded benchmark.
 *
 *  This may look like a trivial example, that may not appear in "real world production systems" of
 * "advanced proprietary software designed by world's leading engineers", but sadly, issues like this
 *  are present in most benchmarks, and sometimes influence multi-billion dollar decisions. 🤬🤬🤬
 *
 *  How bad is it? Let's re-run the same two benchmarks Now, let's run those benchmarks on 8 threads.
 */

BENCHMARK(i32_addition_random)->Threads(8);
BENCHMARK(i32_addition_randomly_initialized)->Threads(8);

/**
 *  The `std::rand` variant latency jumped from @b 100ns in single-threaded mode to @b 12'000ns in multi-threaded,
 *  while our latest variant remained the same.
 *
 *  Like many other LibC functions, it depends on the global state and uses mutexes to synchronize concurrent
 *  access to global memory. Here is its source code in the GNU C Compiler's (GCC) implementation:
 *
 *      long int __random (void) {
 *          int32_t retval;
 *          __libc_lock_lock (lock);
 *          (void) __random_r (&unsafe_state, &retval);
 *          __libc_lock_unlock (lock);
 *          return retval;
 *      }
 *      weak_alias (__random, random)
 *
 *  This is precisely the reason why experienced low-level engineers don't like the "singleton" pattern,
 *  where a single global state is shared between all threads. It's a performance @b killer.
 *
 *  @see GlibC implementation: https://code.woboq.org/userspace/glibc/stdlib/random.c.html#291
 */

#pragma endregion // How to Benchmark and Randomness

#pragma region Parallelism and Computational Complexity

/**
 *  The most obvious approach to speed up the code is to parallelize it. After 2002, the CPUs have mostly
 *  stopped getting faster, and instead, they have been getting wider, with more cores and more threads.
 *  But not all the algorithms can be easily parallelized. Some are inherently sequential, and some are
 *  just too small to benefit from parallelism.
 *
 *  Let's start with @b `std::sort`, probably the best-known and best-implemented algorithm in the C++ standard library.
 *
 *  One option is to random-shuffle on each iteration, but this will make the benchmark less predictable.
 *  Knowing, that the `std::sort` algorithm is based on Quick-Sort, we can reverse the array on each iteration,
 *  which is a well known worst-case scenario for that family of algorithms.
 *
 *  We can also parameterize the benchmark with runtime values, like the size of the array and whether to include
 *  the preprocessing step. This is done with the `Args` function.
 */
static void sorting(bm::State &state) {

    auto count = static_cast<std::size_t>(state.range(0));
    auto include_preprocessing = static_cast<bool>(state.range(1));

    std::vector<std::uint32_t> array(count);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {

        if (!include_preprocessing)
            state.PauseTiming();
        // Reverse order is the most classical worst case, but not the only one.
        std::reverse(array.begin(), array.end());
        if (!include_preprocessing)
            state.ResumeTiming();

        std::sort(array.begin(), array.end());
        bm::DoNotOptimize(array.size());
    }
}

BENCHMARK(sorting)->Args({3, false})->Args({3, true});
BENCHMARK(sorting)->Args({4, false})->Args({4, true});
BENCHMARK(sorting)->Args({1024, false})->Args({1024, true});
BENCHMARK(sorting)->Args({8196, false})->Args({8196, true});

/**
 *  That's the first case, where optimal control flow depends on the input size.
 *  On tiny inputs, its much faster to `include_preprocessing`, while on larger inputs, it's not.
 *
 *  Until C++ 17, the standard didn't have a built-in way to run code in parallel. The C++17 standard
 *  introduced the @b `std::execution` namespace, with the @b `std::execution::par_unseq` policy for
 *  parallel execution of order-independent operations.
 *
 *  The @b `__cpp_lib_parallel_algorithm` macro is one of the feature testing macros, that are defined
 *  by the C++ standards to check, if the parallel algorithms are available:
 * https://en.cppreference.com/w/cpp/utility/feature_test
 */

#if defined(__cpp_lib_parallel_algorithm)

template <typename execution_policy_> static void sorting_with_executors(bm::State &state, execution_policy_ &&policy) {

    auto count = static_cast<std::size_t>(state.range(0));
    std::vector<std::uint32_t> array(count);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {
        std::reverse(policy, array.begin(), array.end());
        std::sort(policy, array.begin(), array.end());
        bm::DoNotOptimize(array.size());
    }

    state.SetComplexityN(count);
    state.SetItemsProcessed(count * state.iterations());
    state.SetBytesProcessed(count * state.iterations() * sizeof(std::int32_t));

    // Feel free to report something else:
    // state.counters["temperature_on_mars"] = bm::Counter(-95.4);
    // But please use the metric system, as opposed to foots and hot-dogs,
    // if you don't want the rockets to crash, like the Mars Climate Orbiter incident of 1999.
}

BENCHMARK_CAPTURE(sorting_with_executors, seq, std::execution::seq)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

BENCHMARK_CAPTURE(sorting_with_executors, par_unseq, std::execution::par_unseq)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

/**
 *  Without @b `UseRealTime()`, CPU time is used by default.
 *  Difference example: when you sleep your process it is no longer accumulating CPU time.
 *
 *  In turn, the @b `Complexity` function is used to specify the asymptotic computational complexity of the benchmark.
 *  To fit the curve, we need to benchmark the function for a fairly broad range of input sizes.
 *  We go from 2^20 (1 Million) to 2^28 entries (256 Million), which is from 4 MB to 1 GB of data.
 *
 *  This would output not just the timings for each input size, but also the inferred complexity:
 *
 *      sorting_with_executors/seq/1048576/min_time:10.000/real_time            5776408 ns      5776186 ns
 *      sorting_with_executors/seq/4194304/min_time:10.000/real_time           25323450 ns     25322993 ns
 *      sorting_with_executors/seq/16777216/min_time:10.000/real_time         109073782 ns    109073030 ns
 *      sorting_with_executors/seq/67108864/min_time:10.000/real_time         482794630 ns    482777617 ns
 *      sorting_with_executors/seq/268435456/min_time:10.000/real_time       2548725384 ns   2548695506 ns
 *      sorting_with_executors/seq/min_time:10.000/real_time_BigO                  0.34 NlgN       0.34 NlgN
 *      sorting_with_executors/seq/min_time:10.000/real_time_RMS                      8 %             8 %
 *
 *  As we can see, with parallel algorithms, the scaling isn't strictly linear, if the task isn't data-parallel.
 */

#endif // defined(__cpp_lib_parallel_algorithm)

#pragma endregion // Parallelism and Computational Complexity

/**
 *  The `std::sort` and the underlying Quick-Sort are perfect research subjects for benchmarking and understanding
 *  how the computer works. Naively implementing the Quick-Sort in C/C++ would still put us at disadvantage, compared
 *  to the STL.
 *
 *  Most implementations we can find in textbooks, use recursion. Recursion is a beautiful concept, but it's not
 *  always the best choice for performance. Every nested call requires a new stack frame, and the stack is limited.
 *  Moreover, local variables need to be constructed and destructed, and the CPU needs to jump around in memory.
 *
 *  The alternative, as it often is in computing, is to use compensate runtime issue with memory. We can use a stack
 *  data structure to continuously store the state of the algorithm, and then process it in a loop.
 *
 *  The same ideas common appear when dealing with trees or graph algorithms.
 */

#pragma region Recursion

/**
 *  @brief  Quick-Sort helper function for array partitioning, reused by both recursive and iterative implementations.
 */
template <typename element_at> struct quick_sort_partition_gt {
    using element_t = element_at;

    std::size_t operator()(element_t *arr, std::size_t low, std::size_t high) noexcept {
        element_t pivot = arr[high];
        std::size_t i = low - 1;
        for (std::size_t j = low; j <= high - 1; j++) {
            if (arr[j] >= pivot)
                continue;
            i++;
            std::swap(arr[i], arr[j]);
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

/**
 *  @brief  Quick-Sort implementation as a C++ function object, using recursion.
 *          Note, recursion and @b inlining are not compatible.
 */
template <typename element_at> //
struct quick_sort_recursive_gt {
    using element_t = element_at;
    using quick_sort_partition_t = quick_sort_partition_gt<element_t>;
    using quick_sort_recursive_t = quick_sort_recursive_gt<element_t>;

    void operator()(element_t *arr, std::size_t low, std::size_t high) noexcept {
        if (low >= high)
            return;
        auto pivot = quick_sort_partition_t{}(arr, low, high);
        quick_sort_recursive_t{}(arr, low, pivot - 1);
        quick_sort_recursive_t{}(arr, pivot + 1, high);
    }
};

/**
 *  @brief  Quick-Sort implementation as a C++ function object, with iterative deepening using
 *          a "stack" data-structure. Note, this implementation can be inlined, but can't be @b `noexcept`,
 *          due to a potential memory allocation in the `std::vector::resize` function.
 */
template <typename element_at> //
struct quick_sort_iterative_gt {
    using element_t = element_at;
    using quick_sort_partition_t = quick_sort_partition_gt<element_t>;

    std::vector<std::size_t> stack;

    inline void operator()(element_t *arr, std::size_t low, std::size_t high) noexcept(false) {

        stack.resize((high - low + 1) * 2);
        std::ptrdiff_t top = -1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];
            auto pivot = quick_sort_partition_t{}(arr, low, high);

            // If there are elements on left side of pivot,
            // then push left side to stack
            if (low < pivot - 1) {
                stack[++top] = low;
                stack[++top] = pivot - 1;
            }

            // If there are elements on right side of pivot,
            // then push right side to stack
            if (pivot + 1 < high) {
                stack[++top] = pivot + 1;
                stack[++top] = high;
            }
        }
    }
};

template <typename sorter_type_, std::size_t length_> //
static void cost_of_recursion(bm::State &state) {
    using element_t = typename sorter_type_::element_t;
    sorter_type_ sorter;
    std::vector<element_t> arr(static_cast<std::size_t>(length_));
    for (auto _ : state) {
        for (std::size_t i = 0; i != length_; ++i)
            arr[i] = length_ - i;
        sorter(arr.data(), 0, length_ - 1);
    }
}

BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_recursive_gt<std::int32_t>, 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_iterative_gt<std::int32_t>, 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_recursive_gt<std::int32_t>, 1024 * 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_iterative_gt<std::int32_t>, 1024 * 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_recursive_gt<std::int32_t>, 1024 * 1024 * 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_iterative_gt<std::int32_t>, 1024 * 1024 * 1024);

#pragma endregion // Recursion

#pragma region Branch Prediction

/**
 *  The `if` statement and seemingly innocent ternary operator (condition ? a : b)
 *  can have a high cost for performance. It's especially noticeable, when conditional
 *  execution is happening at the scale of single bytes, like in text processing,
 *  parsing, search, compression, encoding, and so on.
 *
 *  The CPU has a branch-predictor which is one of the most complex parts of the silicon.
 *  It memorizes the most common `if` statements, to allow "speculative execution".
 *  In other words, start processing the task (i + 1), before finishing the task (i).
 *
 *  Those branch predictors are very powerful, and if you have a single `if` statement
 *  on your hot-path, it's not a big deal. But most programs are almost entirely built
 *  on `if` statements. On most modern CPUs up to 4096 branches will be memorized, but
 *  anything that goes beyond that, would work slower - 3.7 ns vs 0.7 ns for the following
 *  snippet.
 */
static void cost_of_branching_for_different_depth(bm::State &state) {
    auto count = static_cast<std::size_t>(state.range(0));
    std::vector<std::int32_t> random_values(count);
    std::generate_n(random_values.begin(), random_values.size(), &std::rand);
    std::int32_t variable = 0;
    std::size_t iteration = 0;
    for (auto _ : state) {
        std::int32_t random = random_values[(++iteration) & (count - 1)];
        bm::DoNotOptimize(variable = (random & 1) ? (variable + random) : (variable * random));
    }
}

BENCHMARK(cost_of_branching_for_different_depth)->RangeMultiplier(4)->Range(256, 32 * 1024);

#pragma endregion // Branch Prediction

#pragma endregion // - Basics

#pragma region - Numerics

#pragma region Accuracy vs Efficiency of Standard Libraries

/**
 *  Numerics are extensively studied in High-Performance Computing graduate programs and Research Institutes,
 *  but that topic is much more accessible than it seems. Let's start with one of the most basic operations - the sine.
 */

static void f64_sin(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state)
        bm::DoNotOptimize(result = std::sin(argument += 1.0));
}

BENCHMARK(f64_sin);

/**
 *  The `sin` and `sinf` functions are part of the C standard library, and are implemented with maximum accuracy.
 *  This naturally means, there is space for optimization, trading off accuracy for speed.
 *
 *  The conventional approach is to approximate the function with the Taylor-Maclaurin series, which is a polynomial
 *  expansion of a function around a point. Given its an expansion, we can keep a small (and obviously finite) number
 *  of components, and the more we keep, the more accurate the result will be.
 *
 *  To start, we can take 3 components: sin(x) ~ x - (x^3) / 3! + (x^5) / 5!
 *
 *  @see Taylor series: https://en.wikipedia.org/wiki/Taylor_series
 */

static void f64_sin_maclaurin(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = argument - std::pow(argument, 3) / 6 + std::pow(argument, 5) / 120;
        bm::DoNotOptimize(result);
    }
}

BENCHMARK(f64_sin_maclaurin);

/**
 *  The `std::pow` is a general-purpose function, and it's not optimized for the specific case of low integer powers.
 *  We can reimplement the same function with a more specialized version, that will be faster @b and more accurate.
 */

static void f64_sin_maclaurin_powless(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = (argument) - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
}

BENCHMARK(f64_sin_maclaurin_powless);

/**
 *  We want to recommend them to avoid all IEEE-754 compliance checks at a single-function level.
 *  For that, "fast math" attributes can be used: https://simonbyrne.github.io/notes/fastmath/
 *  The problem is, compilers define function attributes in different ways.
 *
 *       -ffast-math (and included by -Ofast) in GCC and Clang
 *       -fp-model=fast (the default) in ICC
 *       /fp:fast in MSVC
 */
#if defined(__GNUC__) && !defined(__clang__)
#define FAST_MATH [[gnu::optimize("-ffast-math")]] //? The old syntax is: __attribute__((optimize("-ffast-math")))
#elif defined(__clang__)
#define FAST_MATH __attribute__((target("-ffast-math")))
#else
#define FAST_MATH
#endif

FAST_MATH static void f64_sin_maclaurin_with_fast_math(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = (argument) - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
}

// Floating point math is not associative!
// So it's not reorderable! And it requires extra annotation!
// Use only when you work with low-mid precision numbers and values of similar magnitude.
// As always with IEEE-754, you have same number of elements in [-inf,-1], [-1,0], [0,1], [1,+inf].
// https://en.wikipedia.org/wiki/Double-precision_floating-point_format
BENCHMARK(f64_sin_maclaurin_with_fast_math);

/**
 *  It's also possible to achieve even higher performance without sacrificing accuracy by using more advanced
 *  procedures, or by reducing the input range. For details check out SimSIMD and SLEEF libraries for different
 *  implementations.
 *
 *  @see SimSIMD repository: https://github.com/ashvardanian/simsimd
 *  @see SLEEF repository: https://github.com/shibatch/sleef
 */

#pragma endregion // Accuracy vs Efficiency of Standard Libraries

#pragma region Expensive Integer Operations

/**
 *  It may be no wonder that complex floating-point operations are expensive, but so can be
 *  a single-instruction integer operations, most famously the @b division and the modulo.
 */

static void integral_division(bm::State &state) {
    std::int64_t a = std::rand() * std::rand(), b = std::rand() * std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / (++b));
}

BENCHMARK(integral_division);

/**
 *  The above operation takes about ~10 CPU cycles or @b 2.5ns.
 *
 *  When the denominator is known at the compile-time, however, the compiler can replace the integer
 *  division with a combination of shifts and multiplications, which is much faster. That can be the
 *  case even with @b prime numbers like the 2147483647.
 *
 *  https://www.sciencedirect.com/science/article/pii/S2405844021015450
 */

static void integral_division_by_constexpr(bm::State &state) {
    constexpr std::int64_t b = 2147483647;
    std::int64_t a = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / b);
}

BENCHMARK(integral_division_by_constexpr);

/**
 *  The @b `constexpr` specifier isn't necessarily needed if the compiler can deduce the same property.
 *  This can affect the benchmarks, but if you want to make sure the true division is used - you can
 *  wrap the variable with `std::launder`
 */

static void integral_division_by_const(bm::State &state) {
    std::int64_t b = 2147483647;
    std::int64_t a = std::rand() * std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / *std::launder(&b));
}

BENCHMARK(integral_division_by_const);

/**
 *  Another important trick to know is that the 32-bit integer division can be expressed accurately
 *  through 64-bit double-precision floating-point division. The latency should go down from 2.5ns
 *  to @b 0.5ns.
 */
static void integral_division_with_doubles(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = static_cast<std::int32_t>(static_cast<double>(++a) / static_cast<double>(++b)));
}

BENCHMARK(integral_division_with_doubles);

/**
 *  It's also crucial to understand that the performance of your code will vary depending on the compilation
 *  settings. The @b `-O3` flag is not enough. Even if you use compiler intrinsics, like the @b `__builtin_popcountll`,
 *  depending on the target CPU generation, it may not have a native Assembly instruction to perform the operation
 *  and will be emulated in software.
 *
 *  We can use GCC attributes to specify the target CPU architecture at a function level. Everything else inside
 *  those functions looks identical, only the flags differ.
 */

#if defined(__GNUC__) && !defined(__clang__)

[[gnu::target("arch=core2")]]
int bits_popcount_emulated(std::uint64_t x) {
    return __builtin_popcountll(x);
}

[[gnu::target("arch=corei7")]]
int bits_popcount_native(std::uint64_t x) {
    return __builtin_popcountll(x);
}

static void bits_population_count_core_2(bm::State &state) {
    auto a = static_cast<std::uint64_t>(std::rand());
    for (auto _ : state)
        bm::DoNotOptimize(bits_popcount_emulated(++a));
}

BENCHMARK(bits_population_count_core_2);

static void bits_population_count_core_i7(bm::State &state) {
    auto a = static_cast<std::uint64_t>(std::rand());
    for (auto _ : state)
        bm::DoNotOptimize(bits_popcount_native(++a));
}

BENCHMARK(bits_population_count_core_i7);
#endif

/**
 *  The difference is @b 3x:
 *  - Core 2 variant: 2.4ns
 *  - Core i7 variant: 0.8ns
 *
 *  Fun fact: there are only a couple of integer operations that can take @b ~100 cycles on select AMD CPUs,
 *  like the BMI2 bit-manipulation instructions, like @b `pdep` and @b `pext`, on AMD Zen 1 and Zen 2 chips.
 *  https://www.chessprogramming.org/BMI2
 */

#pragma endregion // Expensive Integer Operations

/**
 *  Matrix Multiplications are the foundation of Linear Algebra, and are used in a wide range of applications,
 *  including Artificial Intelligence, Computer Graphics, and Physics simulations. Those are so important, that
 *  many CPUs have native instructions for multiplying small matrices, like 4x4 or 8x8. And the larger matrix
 *  multiplications are decomposed into smaller ones, to take advantage of those instructions.
 *
 *  Let's emulate them and learn something new.
 */

#pragma region Compute vs Memory Bounds with Matrix Multiplications

void f32_matrix_multiplication_4x4_loop_kernel(float a[4][4], float b[4][4], float c[4][4]) {
    for (std::size_t i = 0; i != 4; ++i)
        for (std::size_t j = 0; j != 4; ++j) {
            float vector_product = 0;
            for (std::size_t k = 0; k != 4; ++k)
                vector_product += a[i][k] * b[k][j];
            c[i][j] = vector_product;
        }
}

static void f32_matrix_multiplication_4x4_loop(bm::State &state) {
    float a[4][4], b[4][4], c[4][4];
    std::iota(&a[0][0], &a[0][0] + 16, 16);
    std::iota(&b[0][0], &b[0][0] + 16, 0);
    for (auto _ : state) {
        f32_matrix_multiplication_4x4_loop_kernel(a, b, c);
        bm::DoNotOptimize(c);
    }

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32_matrix_multiplication_4x4_loop);

/**
 *  A multiplication of two NxN inputs takes up to NxNxN multiplications and NxNx(N-1) additions.
 *  The asymptote of those operations is O(N^3), and the number of operations grows cubically with
 *  the side of the matrix. The naive kernel above performs those operations in @b 31.5ns.
 *
 *  Most of those operations are data-parallel, so we can probably squeeze more performance.
 *
 *  The most basic trick is loop unrolling. Every @b `for` loop is in reality a @b `goto` and an @b `if`.
 *  As we've learned from the "recursion" and "branching" sections, the jumps and conditions are expensive.
 *  In our case, we explicitly know the the size of the matrix and the number of iterations in every one
 *  of the @b three nested `for` loops. Let's manually express all the operations.
 */

void f32_matrix_multiplication_4x4_loop_unrolled_kernel(float a[4][4], float b[4][4], float c[4][4]) {
    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
    c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
    c[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];

    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
    c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
    c[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];

    c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
    c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
    c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
    c[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];

    c[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0];
    c[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1];
    c[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2];
    c[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3];
}

static void f32_matrix_multiplication_4x4_loop_unrolled(bm::State &state) {
    float a[4][4], b[4][4], c[4][4];
    std::iota(&a[0][0], &a[0][0] + 16, 16);
    std::iota(&b[0][0], &b[0][0] + 16, 0);
    for (auto _ : state) {
        f32_matrix_multiplication_4x4_loop_unrolled_kernel(a, b, c);
        bm::DoNotOptimize(c);
    }

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32_matrix_multiplication_4x4_loop_unrolled);

/**
 *  The unrolled variant executes in @b 11ns, or a @b 3x speedup.
 *
 *  Modern CPUs have a super-scalar execution capability, also called SIMD-computing (Single Instruction, Multiple
 *  Data). It operates on words of 128, 256, or 512 bits, which can contain many 64-, 32-, 16-, or 8-bit components,
 *  like continuous chunks of floats or integers.
 *
 *  Instead of individual scalar operations in the unrolled kernel, let's port to @b SSE4.1 SIMD instructions,
 *  one of the earliest SIMD instruction sets available on most x86 CPUs.
 */

#if defined(__SSE2__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("sse2", "sse3", "sse4.1")
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse2,sse3,sse4.1"))), apply_to = function)
#endif

void f32_matrix_multiplication_4x4_loop_sse41_kernel(float a[4][4], float b[4][4], float c[4][4]) {
    // Load a continuous vector of 4x floats in a single instruction., invoked by the `_mm_loadu_ps` intrinsic.
    __m128 a_row_0 = _mm_loadu_ps(&a[0][0]);
    __m128 a_row_1 = _mm_loadu_ps(&a[1][0]);
    __m128 a_row_2 = _mm_loadu_ps(&a[2][0]);
    __m128 a_row_3 = _mm_loadu_ps(&a[3][0]);

    // Load the columns of the matrix B, by loading the 4 rows and then transposing with an SSE macro:
    // https://randombit.net/bitbashing/posts/integer_matrix_transpose_in_sse2.html
    __m128 b_col_0 = _mm_loadu_ps(&b[0][0]);
    __m128 b_col_1 = _mm_loadu_ps(&b[1][0]);
    __m128 b_col_2 = _mm_loadu_ps(&b[2][0]);
    __m128 b_col_3 = _mm_loadu_ps(&b[3][0]);
    _MM_TRANSPOSE4_PS(b_col_0, b_col_1, b_col_2, b_col_3);

    // Multiply A rows by B columns and store the result in C.
    // Use bitwise "OR" to aggregate dot products and store results.
    //
    // The individual dot products are calculated with the `_mm_dp_ps` intrinsic, which is a dot product
    // of two vectors, with the result stored in a single float. The last argument is a mask, which
    // specifies which components of the vectors should be multiplied and added.
    __m128 c_row_0 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_0, b_col_0, 0xF1), _mm_dp_ps(a_row_0, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_0, b_col_2, 0xF4), _mm_dp_ps(a_row_0, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[0][0], c_row_0);

    __m128 c_row_1 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_1, b_col_0, 0xF1), _mm_dp_ps(a_row_1, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_1, b_col_2, 0xF4), _mm_dp_ps(a_row_1, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[1][0], c_row_1);

    __m128 c_row_2 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_2, b_col_0, 0xF1), _mm_dp_ps(a_row_2, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_2, b_col_2, 0xF4), _mm_dp_ps(a_row_2, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[2][0], c_row_2);

    __m128 c_row_3 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_3, b_col_0, 0xF1), _mm_dp_ps(a_row_3, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_3, b_col_2, 0xF4), _mm_dp_ps(a_row_3, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[3][0], c_row_3);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma clang attribute pop
#endif

static void f32_matrix_multiplication_4x4_loop_sse41(bm::State &state) {
    float a[4][4], b[4][4], c[4][4];
    std::iota(&a[0][0], &a[0][0] + 16, 16);
    std::iota(&b[0][0], &b[0][0] + 16, 0);
    for (auto _ : state) {
        f32_matrix_multiplication_4x4_loop_sse41_kernel(a, b, c);
        bm::DoNotOptimize(c);
    }

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32_matrix_multiplication_4x4_loop_sse41);
#endif // defined(__SSE2__)

/**
 *  The result is @b 18.8ns as opposed to the @b 11.1ns before. Turns out, we were not that smart.
 *  If we disassemble the unrolled kernel, we can see, that the compiler was smart optimizing it.
 *  Each line of that kernel is compiled to a snippet like:
 *
 *      vmovss  xmm0, dword ptr [rdi]
 *      vmovss  xmm1, dword ptr [rdi + 4]
 *      vmulss  xmm1, xmm1, dword ptr [rsi + 16]
 *      vfmadd231ss     xmm1, xmm0, dword ptr [rsi]
 *      vmovss  xmm0, dword ptr [rdi + 8]
 *      vfmadd132ss     xmm0, xmm1, dword ptr [rsi + 32]
 *      vmovss  xmm1, dword ptr [rdi + 12]
 *      vfmadd132ss     xmm1, xmm0, dword ptr [rsi + 48]
 *      vmovss  dword ptr [rdx], xmm1
 *
 *  Seeing the `vfmadd132ss` and `vfmadd231ss` instructions, applied to @b `xmm` registers clearly
 *  indicates, that the compiler was smarter at using SIMD than we are, but the game isn't over.
 *
 *  @see Explore the unrolled kernel assembly on GodBolt: https://godbolt.org/z/bW5nnTKs1
 *
 *  Using AVX-512 on modern CPUs, we can fit an entire matrix in one @b `zmm` register, 512 bits wide.
 *  That Instruction Set Extension is available on Intel Skylake-X, Ice Lake, and AMD Zen4 CPUs, and
 *  has extremely powerful functionality.
 */

#if defined(__AVX512F__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512bw", "avx512vl", "bmi2")
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512bw,avx512vl,bmi2"))), apply_to = function)
#endif

void f32_matrix_multiplication_4x4_loop_avx512_kernel(float a[4][4], float b[4][4], float c[4][4]) {
    __m512 a_mat = _mm512_loadu_ps(&a[0][0]);
    __m512 b_mat = _mm512_loadu_ps(&b[0][0]);
    __m512 c_mat = _mm512_setzero_ps();

    // We need the following intrinsics to implement the matrix multiplication in AVX-512 efficiently:
    //
    // - `_mm512_permutexvar_ps` maps to `vpermps zmm, zmm, zmm`:
    //      - On Intel Skylake-X: 3 cycle latency, port 5.
    //      - On Intel Ice Lake: 3 cycle latency, port 5.
    //      - On AMD Zen4: 6 cycle latency, ports: 1 and 2.
    // - `_mm512_fmadd_ps` maps to `vfmadd231ps zmm, zmm, zmm`:
    //      - On Intel Skylake-X: 4 cycle latency, ports: 0 and 5.
    //      - On Intel Ice Lake: 4 cycle latency, port 0.
    //      - On AMD Zen4: 4 cycle latency, ports: 0 and 1.
    // - `_mm512_mul_ps` maps to `vmulps zmm, zmm, zmm`:
    //      - On Intel Skylake-X: 4 cycle latency, ports: 0 and 5.
    //      - On Intel Ice Lake: 4 cycle latency, port 0.
    //      - On AMD Zen4: 3 cycle latency, ports: 0 and 1.
    // - `_mm512_permute_ps` maps to `vpermilps zmm, zmm, imm8`:
    //      - On Intel Skylake-X: 1 cycle latency, port 5.
    //      - On Intel Ice Lake: 1 cycle latency, port 5.
    //      - On AMD Zen4: 1 cycle latency, ports: 1, 2, 3.
    //
    __m512i b_transposition = _mm512_setr_epi32( //
        15, 11, 7, 3,                            //
        14, 10, 6, 2,                            //
        13, 9, 5, 1,                             //
        12, 8, 4, 0                              //
    );
    b_mat = _mm512_permutexvar_ps(b_transposition, b_mat);

    _mm512_storeu_ps(&c[0][0], c_mat);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma clang attribute pop
#endif

static void f32_matrix_multiplication_4x4_loop_avx512(bm::State &state) {
    float a[4][4], b[4][4], c[4][4];
    std::iota(&a[0][0], &a[0][0] + 16, 16);
    std::iota(&b[0][0], &b[0][0] + 16, 0);
    for (auto _ : state) {
        f32_matrix_multiplication_4x4_loop_avx512_kernel(a, b, c);
        bm::DoNotOptimize(c);
    }

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}
BENCHMARK(f32_matrix_multiplication_4x4_loop_avx512);
#endif // defined(__AVX512F__)

#pragma endregion // Compute vs Memory Bounds with Matrix Multiplications

/**
 *  When composing higher-level kernels from small tiled matrix multiplications, one of the most important
 *  components is tiling the memory and minimizing the number of loads from RAM, maximizing cache utilization.
 *  But not all loads are equal. If you are not careful, you'll end up with unaligned memory addresses, where
 *  part of your data is in one cache line, and the other part is in another.
 *
 *  For large data-structures it's impossible to avoid such "split loads", but for small scalars, it's recommended.
 *  Especially if you are designing a high-performance kernel and you choose the granularity of your processing.
 *
 *  On the majority of modern CPUs the cache line is 64 bytes wide, but its @b not always the case.
 *  On Apple M-series CPUs, for example, the cache line is 128 bytes wide. That's why we can't just put @b `alignas(64)`
 *  everywhere and magically expect total compatibility.
 *
 *  Instead, we need to infer some of those aspects at runtime and invoke kernels designed for different alignments.
 *  To illustrate a point, let's go back to our `std::sort` function and apply it to a buffer of cache-line-sized,
 *  aligned and intentionally misaligned objects, sorted by a single integer.
 *
 *  To implement it, we will:
 *  - design a smart iterator that can access the elements with a given stride (offset multiple in bytes),
 *  - use CRC32 as a cheap, but @b hard-to-predict alternative to increments, to produce a semi-random range to sort,
 *  - use different Operating System APIs to infer the cache line size and the L2 cache size on the current machine,
 *  - flush CPU caches between benchmark iterations, to ensure that the results are not skewed by the cache state.
 */

#pragma region Alignment of Memory Accesses

std::string read_file_contents(std::string const &path) {
    std::ifstream file(path);
    std::string content;
    if (!file.is_open())
        return 0;
    std::getline(file, content);
    file.close();
    return content;
}

/**
 *  Reading memory specs is platform-dependent. It can be done in different ways, like invoking x86 CPUID instructions,
 *  to read the cache line size and the L2 cache size, but even on x86 the exact logic is different between AMD and
 *  Intel. To avoid Assembly, one can use compiler intrinsics, but those differ between compilers and don't provide a
 *  solution for Arm. Alternatively, we use Operating System APIs - different for Linux, MacOS, and Windows.
 */
std::size_t fetch_cache_line_width() {

#if defined(__linux__)
    // On Linux, we can read the cache line size and the L2 cache size from the "sysfs" virtual filesystem.
    // It can provide the properties of each individual CPU core.
    std::string file_contents = read_file_contents("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    std::size_t cache_line_size = std::stoul(file_contents);
    return cache_line_size;

#elif defined(__APPLE__)
    // On macOS, we can use the `sysctlbyname` function to read the `hw.cachelinesize` and `hw.l2cachesize` values
    // into unsigned integers. You can achieve the same by using the `sysctl -a` command-line utility.
    size_t size;
    size_t len = sizeof(size);
    if (sysctlbyname("hw.cachelinesize", &size, &len, nullptr, 0) == 0)
        return size;

#elif defined(_WIN32)
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD len = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &len))
        for (size_t i = 0; i < len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
            if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 1)
                return buffer[i].Cache.LineSize;
#endif

    return 0;
}

/**
 *  We implement a minimalistic strided pointer/iterator, that chooses the next element
 *  address based on the runtime-known variable.
 */
template <typename value_type_> class strided_ptr {
  public:
    using value_type = value_type_;
    using pointer = value_type_ *;
    using reference = value_type_ &;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    inline strided_ptr(std::byte *data, std::size_t stride_bytes) noexcept : data_(data), stride_(stride_bytes) {
        assert(data_ && "Pointer must not be null, as NULL arithmetic is undefined");
    }
    inline reference operator*() const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_)));
    }
    inline reference operator[](difference_type i) const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_ + i * stride_)));
    }

    // clang-format off
    inline pointer operator->() const noexcept { return &operator*(); }
    inline strided_ptr &operator++() noexcept { data_ += stride_; return *this; }
    inline strided_ptr operator++(int) noexcept { strided_ptr temp = *this; ++(*this); return temp; }
    inline strided_ptr &operator--() noexcept { data_ -= stride_; return *this; }
    inline strided_ptr operator--(int) noexcept { strided_ptr temp = *this; --(*this); return temp; }
    inline strided_ptr &operator+=(difference_type offset) noexcept { data_ += offset * stride_; return *this; }
    inline strided_ptr &operator-=(difference_type offset) noexcept { data_ -= offset * stride_; return *this; }
    inline strided_ptr operator+(difference_type offset) noexcept { strided_ptr temp = *this; return temp += offset; }
    inline strided_ptr operator-(difference_type offset) noexcept { strided_ptr temp = *this; return temp -= offset; }
    inline friend difference_type operator-(strided_ptr const &a, strided_ptr const &b) noexcept { assert(a.stride_ == b.stride_); return (a.data_ - b.data_) / static_cast<difference_type>(a.stride_); }
    inline friend bool operator==(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ == b.data_; }
    inline friend bool operator<(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ < b.data_; }
    inline friend bool operator!=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a == b); }
    inline friend bool operator>(strided_ptr const &a, strided_ptr const &b) noexcept { return b < a; }
    inline friend bool operator<=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(b < a); }
    inline friend bool operator>=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a < b); }
    // clang-format on

  private:
    std::byte *data_;
    std::size_t stride_;
};

template <bool aligned> static void memory_access(bm::State &state) {
    constexpr std::size_t typical_l2_size = 1024u * 1024u;
    std::size_t const cache_line_width = fetch_cache_line_width();
    assert(cache_line_width > 0 && __builtin_popcountll(cache_line_width) == 1 &&
           "The cache line width must be a power of two greater than 0");

    // We are using a fairly small L2-cache-sized buffer to show, that this is not just about Big Data.
    // Anything beyond a few megabytes with irregular memory accesses may suffer from the same issues.
    // For split-loads, pad our buffer with an extra `cache_line_width` bytes of space.
    std::size_t const buffer_size = typical_l2_size + cache_line_width;
    std::unique_ptr<std::byte, decltype(&std::free)> const buffer(                        //
        reinterpret_cast<std::byte *>(std::aligned_alloc(cache_line_width, buffer_size)), //
        &std::free);
    std::byte *const buffer_ptr = buffer.get();

    // Let's initialize a strided range using out `strided_ptr` template, but for `aligned == false`
    // make sure that the scalar-of-interest in each stride is located exactly at the boundary between two cache lines.
    std::size_t const offset_within_page = !aligned ? (cache_line_width - sizeof(std::uint32_t) / 2) : 0;
    strided_ptr<std::uint32_t> integers(buffer_ptr + offset_within_page, cache_line_width);

    // We will start with a random seed position and walk through the buffer.
    std::uint32_t semi_random_state = 0xFFFFFFFFu;
    std::size_t const count_pages = typical_l2_size / cache_line_width;
    for (auto _ : state) {
        // Generate some semi-random data, using Knuth's multiplicative hash number derived from the golden ratio.
        std::generate_n(integers, count_pages, [&semi_random_state] { return semi_random_state *= 2654435761u; });

        // Flush all of the pages out of the cache
        // The `__builtin___clear_cache(buffer_ptr, buffer_ptr + buffer.size())`
        // compiler intrinsic can't be used for the data cache, only the instructions cache.
        // For Arm, GCC provides a `__aarch64_sync_cache_range` intrinsic, but it's not available in Clang.
        for (std::size_t i = 0; i != count_pages; ++i)
            _mm_clflush(&integers[i]);
        bm::ClobberMemory();

        std::sort(integers, integers + count_pages);
    }
}

static void memory_access_unaligned(bm::State &state) { memory_access<false>(state); }
static void memory_access_aligned(bm::State &state) { memory_access<true>(state); }

BENCHMARK(memory_access_unaligned)->MinTime(10);
BENCHMARK(memory_access_aligned)->MinTime(10);

/**
 *  One variant executes in 5.8 miliseconds, and the other in 5.2 miliseconds,
 *  consistently resulting a @b 10% performance difference.
 */

#pragma endregion // Alignment of Memory Accesses

#pragma region Non Uniform Memory Access

/**
 *  Takes a string like "64K" and "128M" and returns the corresponding size in bytes,
 *  expanding the multiple prefixes to the actual size, like "65536" and "134217728", respectively.
 */
std::size_t parse_size_string(std::string const &str) {
    std::size_t value = std::stoul(str);
    if (str.find("K") != std::string::npos || str.find("k") != std::string::npos)
        value *= 1024;
    else if (str.find("M") != std::string::npos || str.find("m") != std::string::npos)
        value *= 1024 * 1024;
    else if (str.find("G") != std::string::npos || str.find("g") != std::string::npos)
        value *= 1024 * 1024 * 1024;
    return value;
}

#pragma endregion // Non Uniform Memory Access

#pragma endregion // - Numerics

#pragma region - Abstractions

#pragma region Virtual Functions and Polymorphism

#pragma endregion // Virtual Functions and Polymorphism

#pragma region Ranges and Iterators

#pragma endregion // Ranges and Iterators

#pragma region Coroutines and Asynchronous Programming

#pragma endregion // Coroutines and Asynchronous Programming

#pragma endregion // - Abstractions

/// Calling `std::rand` is clearly expensive, but in some cases we need a semi-random behaviour.
/// A relatively cheap and widely available alternative is to use CRC32 hashes to define the transformation,
/// without pre-computing some random ordering.
///
/// On x86, when SSE4.2 is available, the `crc32` instruction can be used. Both `CRC R32, R/M32` and `CRC32 R32, R/M64`
/// have a latency of 3 cycles on practically all Intel and AMD CPUs,  and can execute only on one port.
/// Check out @b https://uops.info/table for more details.
inline std::uint32_t crc32_hash(std::uint32_t x) noexcept {
#if defined(__SSE4_2__)
    return _mm_crc32_u32(x, 0xFFFFFFFF);
#elif defined(__ARM_FEATURE_CRC32)
    return __crc32cw(x, 0xFFFFFFFF);
#else
    return x * 2654435761u;
#endif
}

/**
 *  The default variant is to invoke the `BENCHMARK_MAIN()` macro.
 *  Alternatively, we can unpack it, if we want to augment the main function
 *  with more system logging logic.
 */

int main(int argc, char **argv) {

    // Let's log the CPU specs:
    std::size_t const cache_line_width = fetch_cache_line_width();
    std::printf("Cache line width: %zu bytes\n", cache_line_width);

    // Make sure the defaults are set correctly:
    char arg0_default[] = "benchmark";
    char *args_default = arg0_default;
    if (!argv)
        argc = 1, argv = &args_default;
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;
    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}