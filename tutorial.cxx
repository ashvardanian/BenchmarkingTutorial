#include <algorithm> // `std::sort`
#include <cmath>     // `std::pow`
#include <cstdint>   // `int32_t`
#include <cstdlib>   // `std::rand`
#include <execution> // `std::execution::par_unseq`
#include <new>       // `std::launder`
#include <random>    // `std::mt19937`
#include <vector>    // `std::algorithm`

#include <benchmark/benchmark.h>

namespace bm = benchmark;

static void i32_addition(bm::State &state) {
    int32_t a = 0, b = 0, c = 0;
    for (auto _ : state)
        c = a + b;

    // Silence "variable ‘c’ set but not used" warning
    (void)c;
}

// The compiler will just optimize everything out.
// After the first run, the value of `c` won't change.
// The benchmark will show 0ns per iteration.
BENCHMARK(i32_addition);

static void i32_addition_random(bm::State &state) {
    int32_t c = 0;
    for (auto _ : state)
        c = std::rand() + std::rand();

    // Silence "variable ‘c’ set but not used" warning
    (void)c;
}

// This run in 25ns, or about 100 CPU cycles.
// Is integer addition really that expensive?
BENCHMARK(i32_addition_random);

static void i32_addition_semi_random(bm::State &state) {
    int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) + (++b));
}

// We trigger the two `inc` instructions and the `add` on x86.
// This shouldn't take more then 0.7 ns on a modern CPU.
// So all the time spent - was in the `rand()`!
BENCHMARK(i32_addition_semi_random);

// Our `rand()` is 100 cycles on a single core, but it involves
// global state management, so it can be as slow 12'000 ns with
// just 8 threads.
BENCHMARK(i32_addition_random)->Threads(8);
BENCHMARK(i32_addition_semi_random)->Threads(8);

// ------------------------------------
// ## Let's do some basic math
// ### Maclaurin series
// ------------------------------------

static void f64_sin(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state)
        bm::DoNotOptimize(result = std::sin(argument += 1.0));
}

static void f64_sin_maclaurin(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = argument - std::pow(argument, 3) / 6 + std::pow(argument, 5) / 120;
        bm::DoNotOptimize(result);
    }
}

// Lets compute the `sin(x)` via Maclaurin series.
// It will involve a fair share of floating point operations.
// We will only take the first 3 parts of the expansion:
//  sin(x) ~ x - (x^3) / 3! + (x^5) / 5!
// https://en.wikipedia.org/wiki/Taylor_series
BENCHMARK(f64_sin);
BENCHMARK(f64_sin_maclaurin);

static void f64_sin_maclaurin_powless(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = argument - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
}

// Help the compiler Help you!
// Instead of using the heavy generic operation - describe your special case to the compiler!
BENCHMARK(f64_sin_maclaurin_powless);

// We want to recommend them to avoid all IEEE-754 compliance checks at a single-function level.
// For that, "fast math" attributes can be used: https://simonbyrne.github.io/notes/fastmath/
// The problem is, compilers define function attributes in different ways.
//      -ffast-math (and included by -Ofast) in GCC and Clang
//      -fp-model=fast (the default) in ICC
//      /fp:fast in MSVC
#if defined(__GNUC__) && !defined(__clang__)
// The old syntax in GCC is: __attribute__((optimize("-ffast-math")))
#define FAST_MATH [[gnu::optimize("-ffast-math")]]
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

// ------------------------------------
// ## Lets look at Integer Division
// ### If floating point arithmetic can be fast, what about integer division?
// ------------------------------------

static void i64_division(bm::State &state) {
    int64_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / (++b));
}

// If we take 32-bit integers - their division can be performed via `double`
// without loss of accuracy. Result: 7ns, or 15x more expensive then addition.
BENCHMARK(i64_division);

static void i64_division_by_const(bm::State &state) {
    int64_t b = 2147483647;
    int64_t a = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / *std::launder(&b));
}

// Let's fix a constant, but `std::launder` it a bit.
// So it looks like a generic pointer and not explicitly
// a constant as a developer might have seen.
// Result: more or less the same as before.
BENCHMARK(i64_division_by_const);

static void i64_division_by_constexpr(bm::State &state) {
    constexpr int64_t b = 2147483647;
    int64_t a = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / b);
}

// But once we mark it as a `constexpr`, the compiler will replace
// heavy divisions with a combination of simpler shifts and multiplications.
// https://www.sciencedirect.com/science/article/pii/S2405844021015450
BENCHMARK(i64_division_by_constexpr);

// ------------------------------------
// ## Where else those tricks are needed
// ------------------------------------
#if defined(__GNUC__) && !defined(__clang__)

[[gnu::target("default")]] static void u64_population_count(bm::State &state) {
    auto a = static_cast<uint64_t>(std::rand());
    for (auto _ : state)
        bm::DoNotOptimize(__builtin_popcount(++a));
}

BENCHMARK(u64_population_count);

[[gnu::target("popcnt")]] static void u64_population_count_x86(bm::State &state) {
    auto a = static_cast<uint64_t>(std::rand());
    for (auto _ : state)
        bm::DoNotOptimize(__builtin_popcount(++a));
}

BENCHMARK(u64_population_count_x86);
#endif

// ------------------------------------
// ## Data Alignment
// ------------------------------------

constexpr size_t f32s_in_cache_line_k = 64 / sizeof(float);
constexpr size_t f32s_in_cache_line_half_k = f32s_in_cache_line_k / 2;

struct alignas(64) f32_array_t {
    float raw[f32s_in_cache_line_k * 2];
};

static void f32_pairwise_accumulation(bm::State &state) {
    f32_array_t a, b, c;
    for (auto _ : state)
        for (size_t i = f32s_in_cache_line_half_k; i != f32s_in_cache_line_half_k * 3; ++i)
            bm::DoNotOptimize(c.raw[i] = a.raw[i] + b.raw[i]);
}

static void f32_pairwise_accumulation_aligned(bm::State &state) {
    f32_array_t a, b, c;
    for (auto _ : state)
        for (size_t i = 0; i != f32s_in_cache_line_half_k; ++i)
            bm::DoNotOptimize(c.raw[i] = a.raw[i] + b.raw[i]);
}

// Split load occurs in the first case and doesn't in the second.
// We do the same number of arithmetical operations, but:
//      - first takes 8 ns
//      - second takes 4 ns
BENCHMARK(f32_pairwise_accumulation)->MinTime(10);
BENCHMARK(f32_pairwise_accumulation_aligned)->MinTime(10);

// ------------------------------------
// ## Cost of Control Flow
// ------------------------------------

// The `if` statement and seemingly innocent ternary operator (condition ? a : b)
// Can have a high for performance. It's especially noticeable, when conditional
// execution is happening at the scale of single bytes, like in text processing,
// parsing, search, compression, encoding, and so on.
//
// The CPU has a branch-predictor which is one of the most complex parts of the silicon.
// It memorizes the most common `if` statements, to allow "speculative execution".
// In other words, start processing the task (i + 1), before finishing the task (i).
//
// Those branch predictors are very powerful, and if you have a single `if` statement
// on your hot-path, it's not a big deal. But most programs are almost entirely built
// on `if` statements. On most modern CPUs up to 4096 branches will be memorized, but
// anything that goes beyond that, would work slower - 2.9 ns vs 0.7 ns for the following snippet.
static void cost_of_branching_for_different_depth(bm::State &state) {
    auto count = static_cast<size_t>(state.range(0));
    std::vector<int32_t> rands(count);
    std::generate_n(rands.begin(), rands.size(), &std::rand);
    int32_t c = 0;
    size_t i = 0;
    for (auto _ : state) {
        int32_t r = rands[(++i) & (count - 1)];
        bm::DoNotOptimize(c = (r & 1) ? (c + r) : (c * r));
    }
}

BENCHMARK(cost_of_branching_for_different_depth)->RangeMultiplier(4)->Range(128, 32 * 1024);

// We don't have to generate a large array of random numbers to showcase the cost of branching.
// Simple one-line statement can be enough to cause the same 2.2 ns slowdown.
static void cost_of_branching_without_random_arrays(bm::State &state) {
    int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (c & 1) ? ((a--) + (b)) : ((++b) - (a)));
}

BENCHMARK(cost_of_branching_without_random_arrays);

// Google Benchmark also provides it's own Control Flow primitives, to control timing.
// Those `PauseTiming` and `ResumeTiming` functions, however, are not free.
// In current implementation, they can easily take ~127 ns, or around 300 CPU cycles.
static void cost_of_pausing(bm::State &state) {
    int32_t a = std::rand(), c = 0;
    for (auto _ : state) {
        state.PauseTiming();
        ++a;
        state.ResumeTiming();
        bm::DoNotOptimize(c += a);
    }
}

BENCHMARK(cost_of_pausing);

// ------------------------------------
// ## Bulk Operations
// ------------------------------------

static void sorting(bm::State &state) {

    auto count = static_cast<size_t>(state.range(0));
    auto include_preprocessing = static_cast<bool>(state.range(1));

    std::vector<int32_t> array(count);
    std::iota(array.begin(), array.end(), 1);

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

// `std::sort` will invoke a modification of Quick-Sort.
// It's worst case complexity is ~O(N^2), but what the hell are those numbers??
BENCHMARK(sorting)->Args({3, false})->Args({3, true});
BENCHMARK(sorting)->Args({4, false})->Args({4, true});

template <bool include_preprocessing_k> static void sorting_template(bm::State &state) {

    auto count = static_cast<size_t>(state.range(0));
    std::vector<int32_t> array(count);
    std::iota(array.begin(), array.end(), 1);

    for (auto _ : state) {

        if constexpr (!include_preprocessing_k)
            state.PauseTiming();
        std::reverse(array.begin(), array.end());
        if constexpr (!include_preprocessing_k)
            state.ResumeTiming();

        std::sort(array.begin(), array.end());
        bm::DoNotOptimize(array.size());
    }
}

// Now, our control-flow will not affect the measurements!
// "Don't pay what you don't use" becomes: "Don't pay for what you can avoid!"
BENCHMARK_TEMPLATE(sorting_template, false)->Arg(3);
BENCHMARK_TEMPLATE(sorting_template, true)->Arg(3);
BENCHMARK_TEMPLATE(sorting_template, false)->Arg(4);
BENCHMARK_TEMPLATE(sorting_template, true)->Arg(4);

template <typename element_at> //
struct quick_sort_partition_gt {
    using element_t = element_at;

    std::int32_t operator()(element_t *arr, std::int32_t low, std::int32_t high) {
        element_t pivot = arr[high];
        std::int32_t i = low - 1;
        for (std::int32_t j = low; j <= high - 1; j++) {
            if (arr[j] >= pivot)
                continue;
            i++;
            std::swap(arr[i], arr[j]);
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

template <typename element_at> //
struct quick_sort_recursive_gt {
    using element_t = element_at;
    using quick_sort_partition_t = quick_sort_partition_gt<element_t>;
    using quick_sort_recursive_t = quick_sort_recursive_gt<element_t>;

    void operator()(element_t *arr, std::int32_t low, std::int32_t high) {
        if (low >= high)
            return;
        auto pivot = quick_sort_partition_t{}(arr, low, high);
        quick_sort_recursive_t{}(arr, low, pivot - 1);
        quick_sort_recursive_t{}(arr, pivot + 1, high);
    }
};

template <typename element_at> //
struct quick_sort_iterative_gt {
    using element_t = element_at;
    using quick_sort_partition_t = quick_sort_partition_gt<element_t>;

    std::vector<std::int32_t> stack;

    void operator()(element_t *arr, std::int32_t low, std::int32_t high) {

        stack.resize((high - low + 1) * 2);
        std::int32_t top = -1;

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

template <typename sorter_at, std::int32_t length_ak> //
static void cost_of_recursion(bm::State &state) {
    using element_t = typename sorter_at::element_t;
    sorter_at sorter;
    std::vector<element_t> arr(static_cast<std::size_t>(length_ak));
    for (auto _ : state) {
        for (std::int32_t i = 0; i != length_ak; ++i)
            arr[i] = length_ak - i;
        sorter(arr.data(), 0, length_ak - 1);
    }
}

BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_recursive_gt<std::int32_t>, 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_iterative_gt<std::int32_t>, 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_recursive_gt<std::int32_t>, 1024 * 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_iterative_gt<std::int32_t>, 1024 * 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_recursive_gt<std::int32_t>, 1024 * 1024 * 1024);
BENCHMARK_TEMPLATE(cost_of_recursion, quick_sort_iterative_gt<std::int32_t>, 1024 * 1024 * 1024);

// ------------------------------------
// ## Now that we know how fast algorithm works - lets scale it!
// ### And learn the rest of relevant functionality in the process
// ------------------------------------

template <typename execution_policy_t> static void super_sort(bm::State &state, execution_policy_t &&policy) {

    auto count = static_cast<size_t>(state.range(0));
    std::vector<int32_t> array(count);
    std::iota(array.begin(), array.end(), 1);

    for (auto _ : state) {
        std::reverse(policy, array.begin(), array.end());
        std::sort(policy, array.begin(), array.end());
        bm::DoNotOptimize(array.size());
    }

    state.SetComplexityN(count);
    state.SetItemsProcessed(count * state.iterations());
    state.SetBytesProcessed(count * state.iterations() * sizeof(int32_t));

    // Feel free to report something else:
    // state.counters["temperature_on_mars"] = bm::Counter(-95.4);
}

#ifdef __cpp_lib_parallel_algorithm

// Let's try running on 1M to 4B entries.
// This means input sizes between 4 MB and 16 GB respectively.
BENCHMARK_CAPTURE(super_sort, seq, std::execution::seq)
    ->RangeMultiplier(8)
    ->Range(1l << 20, 1l << 32)
    ->MinTime(10)
    ->Complexity(bm::oNLogN);

BENCHMARK_CAPTURE(super_sort, par_unseq, std::execution::par_unseq)
    ->RangeMultiplier(8)
    ->Range(1l << 20, 1l << 32)
    ->MinTime(10)
    ->Complexity(bm::oNLogN);

// Without `UseRealTime()`, CPU time is used by default.
// Difference example: when you sleep your process it is no longer accumulating CPU time.
BENCHMARK_CAPTURE(super_sort, par_unseq, std::execution::par_unseq)
    ->RangeMultiplier(8)
    ->Range(1l << 20, 1l << 32)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

#endif

// ------------------------------------
// ## Practical Investigation Example
// ------------------------------------

BENCHMARK_MAIN();
