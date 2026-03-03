"""Shared pytest configuration for pbc_datagen tests."""

import os

# Cap OpenMP threads to 4 for all tests.  Without this, OpenMP defaults
# to all cores (e.g. 22) which causes catastrophic slowdown on small
# workloads due to cache thrashing and memory bandwidth saturation.
# Must be set before the first C++ import triggers thread pool creation.
os.environ.setdefault("OMP_NUM_THREADS", "4")
