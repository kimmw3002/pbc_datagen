cmake_minimum_required(VERSION 3.20)
project(pbc_datagen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization for inner loops (-ffast-math deliberately omitted: it breaks
# floating-point guarantees that Monte Carlo detailed balance depends on)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_core
    src/cpp/ising.cpp
    src/cpp/blume_capel.cpp
    src/cpp/ashkin_teller.cpp
    src/cpp/bindings.cpp
)

target_include_directories(_core PRIVATE src/cpp/include)

# Link OpenMP — enables #pragma omp in PT sweep loops.
# find_package(OpenMP) is tried first (portable). If cmake's detection fails
# (common with scikit-build-core on some Linux setups), fall back to the GCC
# -fopenmp flag directly, which links libgomp automatically.
# FATAL_ERROR if neither works: a silently single-threaded binary is worse
# than a build failure because the bug is invisible at runtime.
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found via find_package — linking OpenMP::OpenMP_CXX")
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "find_package(OpenMP) failed — applying -fopenmp directly (GCC/Clang fallback)")
    target_compile_options(_core PRIVATE -fopenmp)
    target_link_options(_core PRIVATE -fopenmp)
else()
    message(FATAL_ERROR
        "OpenMP not found and no -fopenmp fallback available for compiler '${CMAKE_CXX_COMPILER_ID}'.\n"
        "Install libomp-dev (apt) or equivalent, then rebuild.")
endif()

install(TARGETS _core DESTINATION pbc_datagen)
