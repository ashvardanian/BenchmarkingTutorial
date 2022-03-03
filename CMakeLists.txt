cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
include(FetchContent)

# Fetch GTest, as its needed for GBenchmark
# https://stackoverflow.com/a/60417587
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# Fetch GBenchmark and surpress internal tests.
# https://github.com/google/benchmark/blob/main/docs/user_guide.md#using-register-benchmark
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.6.1  
)
set(BENCHMARK_ENABLE_TESTING off)
FetchContent_GetProperties(googlebenchmark)
if(NOT googlebenchmark_POPULATED)
    FetchContent_Populate(googlebenchmark)
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

# We need TBB for Parallel CPU Algorithms in GCC.
# https://github.com/oneapi-src/oneTBB/blob/onetbb_2021/cmake/README.md
FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG        v2021.5.0
)
FetchContent_Populate(TBB)
if(NOT TBB_POPULATED)
    include_directories(BEFORE ${TBB_SOURCE_DIR})
endif()


# Define the supported languages
project(benchmarking_tutorial LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Make Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3") # Fast by default :)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unused-but-set-variable") # Less warnings
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -falign-functions=32") # Stable results between recompilations

# Add dependencies
find_package(Threads REQUIRED)
find_package(OpenMP)
find_package(TBB)

add_executable(main main.cxx)
set_target_properties(main PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(main 
    benchmark::benchmark
    Threads::Threads
    TBB::tbb
)