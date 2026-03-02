# pbc_datagen — Paper-quality 2D lattice model snapshot generator.

from loguru import logger

# Disable logging by default so demo scripts and benchmarks stay silent.
# The CLI (scripts/generate_dataset.py) calls logger.enable("pbc_datagen")
# after configuring its own handlers.
logger.disable("pbc_datagen")
