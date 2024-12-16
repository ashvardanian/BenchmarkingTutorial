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
#include <new>       // `std::launder`
#include <random>    // `std::mt19937`
#include <vector>    // `std::algorithm`

#include <benchmark/benchmark.h>

namespace bm = benchmark;

static void i32_addition(bm::State &state) {
    std::int32_t a = 0, b = 0, c = 0;
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
    std::int32_t c = 0;
    for (auto _ : state)
        c = std::rand() + std::rand();

    // Silence "variable ‘c’ set but not used" warning
    (void)c;
}

// This run in 25ns, or about 100 CPU cycles.
// Is integer addition really that expensive?
BENCHMARK(i32_addition_random);

static void i32_addition_semi_random(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
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
    std::int64_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / (++b));
}

// If we take 32-bit integers - their division can be performed via `double`
// without loss of accuracy. Result: 7ns, or 15x more expensive then addition.
BENCHMARK(i64_division);

static void i64_division_by_const(bm::State &state) {
    std::int64_t b = 2147483647;
    std::int64_t a = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (++a) / *std::launder(&b));
}

// Let's fix a constant, but `std::launder` it a bit.
// So it looks like a generic pointer and not explicitly
// a constant as a developer might have seen.
// Result: more or less the same as before.
BENCHMARK(i64_division_by_const);

static void i64_division_by_constexpr(bm::State &state) {
    constexpr std::int64_t b = 2147483647;
    std::int64_t a = std::rand(), c = 0;
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

struct memory_specs_t {
    std::size_t l2_cache_size = 1024 * 1024; ///< Default to 1MB
    std::size_t cache_line_size = 64;        ///< Default to 64 bytes
};

std::size_t parse_size_string(std::string const &str) {
    std::size_t value = std::stoul(str);
    if (str.find("K") != std::string::npos || str.find("k") != std::string::npos) {
        value *= 1024;
    } else if (str.find("M") != std::string::npos || str.find("m") != std::string::npos) {
        value *= 1024 * 1024;
    }
    return value;
}

std::size_t read_file_contents(std::string const &path) {
    std::ifstream file(path);
    std::string content;
    if (file.is_open()) {
        std::getline(file, content);
        file.close();
        return parse_size_string(content);
    }
    return 0;
}

memory_specs_t fetch_memory_specs() {
    memory_specs_t specs;

#if defined(__linux__)
    specs.cache_line_size = read_file_contents("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    specs.l2_cache_size = read_file_contents("/sys/devices/system/cpu/cpu0/cache/index2/size");

#elif defined(__APPLE__)
    size_t size;
    size_t len = sizeof(size);
    if (sysctlbyname("hw.cachelinesize", &size, &len, nullptr, 0) == 0) {
        specs.cache_line_size = size;
    }
    if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0) {
        specs.l2_cache_size = size;
    }

#elif defined(_WIN32)
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD len = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &len)) {
        for (size_t i = 0; i < len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i) {
            if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 2) {
                specs.l2_cache_size = buffer[i].Cache.Size;
            }
            if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 1) {
                specs.cache_line_size = buffer[i].Cache.LineSize;
            }
        }
    }
#endif

    return specs;
}

/// Calling `std::rand` is clearly expensive, but in some cases we need a semi-random behaviour.
/// A relatively cheap and widely available alternative is to use CRC32 hashes to define the transformation,
/// without pre-computing some random ordering.
///
/// On x86, when SSE4.2 is available, the `crc32` instruction can be used. Both `CRC R32, R/M32` and `CRC32 R32, R/M64`
/// have a latency of 3 cycles on practically all Intel and AMD CPUs,  and can execute only on one port.
/// Check out @b https://uops.info/table for more details.
inline std::uint32_t crc32_hash(std::uint32_t x) noexcept {
    return x * 2654435761u;
#if defined(__SSE4_2__)
    return _mm_crc32_u32(x, 0xFFFFFFFF);
#elif defined(__ARM_FEATURE_CRC32)
    return __crc32cw(x, 0xFFFFFFFF);
#else
    return x * 2654435761u;
#endif
}

inline void fetch_64_bytes(std::byte *bytes) noexcept {
    std::uint64_t *words = reinterpret_cast<std::uint64_t *>(bytes);
    bm::DoNotOptimize(words[0] + words[7]);
    bm::DoNotOptimize(words[1] + words[6]);
    bm::DoNotOptimize(words[2] + words[5]);
    bm::DoNotOptimize(words[3] + words[4]);
}

template <typename value_type_> class strided_iterator {
  public:
    using value_type = value_type_;
    using pointer = value_type_ *;
    using reference = value_type_ &;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    inline strided_iterator(std::byte *byte_ptr, std::size_t stride_bytes) noexcept
        : byte_ptr_(byte_ptr), stride_bytes_(stride_bytes) {
        assert(byte_ptr_ && "Pointer must not be null");
    }

    inline reference operator[](difference_type index) const noexcept {
        return *reinterpret_cast<pointer>(byte_ptr_ + index * stride_bytes_);
    }

    inline reference operator*() const noexcept { return operator[](0); }
    inline pointer operator->() const noexcept { return &operator*(); }

    inline strided_iterator &operator++() noexcept {
        byte_ptr_ += stride_bytes_;
        return *this;
    }

    inline strided_iterator operator++(int) noexcept {
        strided_iterator temp = *this;
        ++(*this);
        return temp;
    }

    inline strided_iterator &operator--() noexcept {
        byte_ptr_ -= stride_bytes_;
        return *this;
    }

    inline strided_iterator operator--(int) noexcept {
        strided_iterator temp = *this;
        --(*this);
        return temp;
    }

    inline strided_iterator &operator+=(difference_type offset) noexcept {
        byte_ptr_ += offset * stride_bytes_;
        return *this;
    }

    inline strided_iterator &operator-=(difference_type offset) noexcept {
        byte_ptr_ -= offset * stride_bytes_;
        return *this;
    }

    inline strided_iterator operator+(difference_type offset) noexcept {
        strided_iterator temp = *this;
        return temp += offset;
    }
    inline strided_iterator operator-(difference_type offset) noexcept {
        strided_iterator temp = *this;
        return temp -= offset;
    }

    inline friend difference_type operator-(strided_iterator const &a, strided_iterator const &b) noexcept {
        assert(a.stride_bytes_ == b.stride_bytes_);
        return (a.byte_ptr_ - b.byte_ptr_) / static_cast<difference_type>(a.stride_bytes_);
    }

    inline friend bool operator==(strided_iterator const &a, strided_iterator const &b) noexcept {
        return a.byte_ptr_ == b.byte_ptr_;
    }
    inline friend bool operator<(strided_iterator const &a, strided_iterator const &b) noexcept {
        return a.byte_ptr_ < b.byte_ptr_;
    }

    inline friend bool operator!=(strided_iterator const &a, strided_iterator const &b) noexcept { return !(a == b); }
    inline friend bool operator>(strided_iterator const &a, strided_iterator const &b) noexcept { return b < a; }
    inline friend bool operator<=(strided_iterator const &a, strided_iterator const &b) noexcept { return !(b < a); }
    inline friend bool operator>=(strided_iterator const &a, strided_iterator const &b) noexcept { return !(a < b); }

  private:
    std::byte *byte_ptr_;
    std::size_t stride_bytes_;
};

template <bool aligned> static void memory_access(bm::State &state) {
    memory_specs_t const memory_specs = fetch_memory_specs();
    assert(                                                      //
        memory_specs.l2_cache_size > 0 &&                        //
        __builtin_popcount(memory_specs.l2_cache_size) == 1 &&   //
        __builtin_popcount(memory_specs.cache_line_size) == 1 && //
        "L2 cache size and cache line width must be a power of two greater than 0");

    std::size_t const l2_buffer_size = memory_specs.l2_cache_size + memory_specs.cache_line_size;
    std::unique_ptr<std::byte[]> const l2_buffer = std::make_unique<std::byte[]>(l2_buffer_size);
    std::byte *const l2_buffer_ptr = l2_buffer.get();

    std::size_t const offset_within_page = !aligned ? memory_specs.cache_line_size - sizeof(std::uint32_t) / 2 : 0;
    strided_iterator<std::uint32_t> integers(l2_buffer_ptr + offset_within_page, memory_specs.cache_line_size);

    // We will start with a random seed position and walk through the buffer.
    std::uint32_t semi_random_state = std::rand();
    std::size_t const count_pages = memory_specs.l2_cache_size / memory_specs.cache_line_size;
    std::size_t const count_cycles = count_pages / 8;
    for (auto _ : state) {
        // Generate some semi-random data
        std::generate_n(integers, count_pages,
                        [&semi_random_state] { return semi_random_state = crc32_hash(semi_random_state); });

        // Flush all of the pages out of the cache
        // The `__builtin___clear_cache(l2_buffer_ptr, l2_buffer_ptr + l2_buffer.size())`
        // compiler intrinsic can't be used for the data cache, only the instructions cache.
        for (std::size_t i = 0; i != count_pages; ++i)
            _mm_clflush(l2_buffer_ptr + i * memory_specs.cache_line_size);
        bm::ClobberMemory();

        std::sort(integers, integers + count_pages);
    }
}

static void memory_access_unaligned(bm::State &state) { memory_access<false>(state); }
static void memory_access_aligned(bm::State &state) { memory_access<true>(state); }

// Split load occurs in the second case but not in the first.
// While the number of access operations is the same,
// crossing 64-byte cache-line boundaries can be significantly slower.
BENCHMARK(memory_access_unaligned)->MinTime(10);
BENCHMARK(memory_access_aligned)->MinTime(10);

// ------------------------------------
// ## Cost of Control Flow
// ------------------------------------

// The `if` statement and seemingly innocent ternary operator (condition ? a : b)
// can have a high for performance. It's especially noticeable, when conditional
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

// We don't have to generate a large array of random numbers to showcase the cost of branching.
// Simple one-line statement can be enough to cause the same 2.2 ns slowdown.
static void cost_of_branching_without_random_arrays(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = (c & 1) ? ((a--) + (b)) : ((++b) - (a)));
}

BENCHMARK(cost_of_branching_without_random_arrays);

// Google Benchmark also provides it's own Control Flow primitives, to control timing.
// Those `PauseTiming` and `ResumeTiming` functions, however, are not free.
// In current implementation, they can easily take ~127 ns, or around 300 CPU cycles.
static void cost_of_pausing(bm::State &state) {
    std::int32_t a = std::rand(), c = 0;
    for (auto _ : state) {
        state.PauseTiming();
        ++a;
        state.ResumeTiming();
        bm::DoNotOptimize(c += a);
    }
}

BENCHMARK(cost_of_pausing);

// ------------------------------------
// ## Loop Unrolling
// ------------------------------------

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

#if defined(__SSE2__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("sse2", "sse3", "sse4.1")
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse2,sse3,sse4.1"))), apply_to = function)
#endif

void f32_matrix_multiplication_4x4_loop_sse41_kernel(float a[4][4], float b[4][4], float c[4][4]) {
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
    // Use OR to aggregate dot products and store results
    __m128 c_row_0 = _mm_or_ps(                //
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_0, b_col_0, 0xF1), //
            _mm_dp_ps(a_row_0, b_col_1, 0xF2)),
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_0, b_col_2, 0xF4), //
            _mm_dp_ps(a_row_0, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[0][0], c_row_0);

    __m128 c_row_1 = _mm_or_ps(                //
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_1, b_col_0, 0xF1), //
            _mm_dp_ps(a_row_1, b_col_1, 0xF2)),
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_1, b_col_2, 0xF4), //
            _mm_dp_ps(a_row_1, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[1][0], c_row_1);

    __m128 c_row_2 = _mm_or_ps(                //
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_2, b_col_0, 0xF1), //
            _mm_dp_ps(a_row_2, b_col_1, 0xF2)),
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_2, b_col_2, 0xF4), //
            _mm_dp_ps(a_row_2, b_col_3, 0xF8)));
    _mm_storeu_ps(&c[2][0], c_row_2);

    __m128 c_row_3 = _mm_or_ps(                //
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_3, b_col_0, 0xF1), //
            _mm_dp_ps(a_row_3, b_col_1, 0xF2)),
        _mm_or_ps(                             //
            _mm_dp_ps(a_row_3, b_col_2, 0xF4), //
            _mm_dp_ps(a_row_3, b_col_3, 0xF8)));
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
#endif // defined(__SSE2__)

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
#endif // defined(__AVX512F__)

BENCHMARK(f32_matrix_multiplication_4x4_loop);
BENCHMARK(f32_matrix_multiplication_4x4_loop_unrolled);
#if defined(__SSE2__)
BENCHMARK(f32_matrix_multiplication_4x4_loop_sse41);
#endif
#if defined(__AVX512F__)
BENCHMARK(f32_matrix_multiplication_4x4_loop_avx512);
#endif

// ------------------------------------
// ## Bulk Operations
// ------------------------------------

static void sorting(bm::State &state) {

    auto count = static_cast<std::size_t>(state.range(0));
    auto include_preprocessing = static_cast<bool>(state.range(1));

    std::vector<std::int32_t> array(count);
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

    auto count = static_cast<std::size_t>(state.range(0));
    std::vector<std::int32_t> array(count);
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

    auto count = static_cast<std::size_t>(state.range(0));
    std::vector<std::int32_t> array(count);
    std::iota(array.begin(), array.end(), 1);

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
// ## Calling the benchmarks
// ------------------------------------
//
// The default variant is to invoke the `BENCHMARK_MAIN()` macro.
// Alternatively, we can unpack it, if we want to augment the main function
// with more system logging logic.
int main(int argc, char **argv) {

    // Let's log the CPU specs:
    memory_specs_t const specs = fetch_memory_specs();
    std::printf("Cache Line Size: %zu bytes\n", specs.cache_line_size);
    std::printf("L2 Cache Size: %zu bytes\n", specs.l2_cache_size);

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