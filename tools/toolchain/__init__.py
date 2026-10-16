"""The compiler build recipes, shared by run_tests.py and CI.

clang.py and gcc.py each build one reflection compiler; common.py holds what
they and run_tests.py share. Run a recipe from the repo root with
`python3 -m tools.toolchain.clang` (or `.gcc`); --help lists its options.
"""
