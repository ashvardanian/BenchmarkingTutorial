cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
include(FetchContent)

# Fetch GTest, as its needed for GBenchmark https://stackoverflow.com/a/60417587
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

# Fetch GBenchmark and surpress internal tests.
# https://github.com/google/benchmark/blob/main/docs/user_guide.md#using-register-benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(BENCHMARK_ENABLE_LIBPFM ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googlebenchmark)

# We need TBB for Parallel CPU Algorithms in GCC.
# https://github.com/oneapi-src/oneTBB/blob/onetbb_2021/cmake/README.md
FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2021.11.0)

# Set the TBB_TEST option to OFF before making it available
set(TBB_TEST OFF CACHE BOOL "Do not build TBB tests" FORCE)
FetchContent_MakeAvailable(TBB)

# Define the supported languages
project(tutorial LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Make Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Fast by default :)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Less warnings
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unused-but-set-variable")

# Stable results between recompilations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -falign-functions=32")

# Add dependencies
find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(tutorial tutorial.cxx)
set_target_properties(tutorial PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tutorial pthread benchmark)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(tutorial TBB::tbb)
endif()
