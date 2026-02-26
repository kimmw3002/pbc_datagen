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

pybind11_add_module(_core
    src/cpp/ising.cpp
    src/cpp/blume_capel.cpp
    src/cpp/ashkin_teller.cpp
    src/cpp/bindings.cpp
)

target_include_directories(_core PRIVATE src/cpp/include)

install(TARGETS _core DESTINATION pbc_datagen)
