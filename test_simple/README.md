# Compilation Failure Testing

This directory contains a snapshot-based testing system for validating expected compilation failures. Tests verify not just that code fails to compile, but that it fails for the *correct reason*.

## Overview

Compilation failure tests ensure that static assertions, type errors, and other compile-time diagnostics catch bugs as intended. Without snapshot testing, a typo in the error message could go undetected.

**Key features:**
- Semantic error comparison (strips paths, line numbers, formatting noise)
- Multi-compiler support: single `.fail.txt` file tracks expected errors for Clang and GCC
- Rebase workflow: easily update expected output when intentional changes are made
- Regression detection: catches accidental changes to error messages

## File Structure

```
test_simple/
├── README.md                          (this file)
├── test_simple_cmake.cmake            (CMake functions for test registration)
├── test_simple_include.hpp            (test macros: EXPECT_EQUAL, EXPECT_NEAR_ABS, etc.)
├── shared_extract.cmake               (common extraction and normalization functions)
├── validate_compile_output.cmake      (test validation script)
└── extract_error_output.cmake         (baseline extraction script)

tests/
└── static_fail/
    ├── inverse_of_non_invertible.cpp
    └── inverse_of_non_invertible.fail.txt
```

The `.fail.txt` file lives next to the `.cpp` file and contains expected compilation errors for all supported compilers.

## Workflow

### 1. Create a Failing Compilation Test

```cmake
# In CMakeLists.txt
compile_check(reflection_ad "static_fail/inverse_of_non_invertible.cpp")
```

### 2. Create the First Baseline

When a test runs for the first time with no `.fail.txt` file, it fails with:
```
Expected output file not found: .../inverse_of_non_invertible.fail.txt

This is the first time this test is being run. To create the baseline:
  cmake --build . --target rebase-reflection_ad.static_fail.inverse_of_non_invertible
```

Run the rebase command:
```bash
cmake --build build/cmake --target rebase-reflection_ad.static_fail.inverse_of_non_invertible
```

This:
- Attempts to compile the target
- Extracts error/warning lines
- Adds a compiler header (`=== Clang ===` or `=== GCC ===`)
- Writes to `.fail.txt`

### 3. Review and Commit

```bash
git diff tests/static_fail/inverse_of_non_invertible.fail.txt
git add tests/static_fail/inverse_of_non_invertible.fail.txt
```

### 4. Run Tests

```bash
ctest --test-dir build/cmake -R inverse_of_non_invertible --output-on-failure
```

Expected output:
```
1/1 Test #16: reflection_ad.static_fail.inverse_of_non_invertible ...   Passed
100% tests passed, 0 tests failed out of 1
```

## Multi-Compiler Support

A single `.fail.txt` file contains expected errors for multiple compilers:

```
=== Clang ===
inverse_of_non_invertible.cpp:42:3: error: static assertion failed: ...
  ^ ~~
1 error generated.

=== GCC ===
inverse_of_non_invertible.cpp:42:3: error: static assertion failed
42 |   static_assert(false);
   |   ^~~~~~~~~~~~~~
```

### Rebasing with a Specific Compiler

```bash
# Switch to Clang CMake session
cmake --build build/cmake --target rebase-reflection_ad.static_fail.inverse_of_non_invertible

# Switch to GCC CMake session
cmake --build build/cmake --target rebase-reflection_ad.static_fail.inverse_of_non_invertible
```

Each rebase:
- Preserves other compilers' sections
- Updates only the current compiler's section
- Maintains `.fail.txt` format

### Running Tests with a Specific Compiler

Tests automatically validate against the current compiler's section:

```bash
# With Clang CMake session active
ctest --test-dir build/cmake -R inverse_of_non_invertible

# With GCC CMake session active
ctest --test-dir build/cmake -R inverse_of_non_invertible
```

## Comparison Process

The validation script:
1. Extracts error lines from actual build output
2. Reads the expected baseline (saved as-is in `.fail.txt`)
3. **Normalizes both** for comparison (strips paths, line numbers, whitespace noise)
4. Compares normalized versions

The `.fail.txt` baseline stores the **full, original error output** (including paths and line numbers). Normalization is applied only during comparison, making tests resilient to:

- **Path variations**: `/path/to/file.cpp:1:2: error:` vs `/other/path/file.cpp:1:2: error:`
- **Whitespace differences**: Multiple spaces collapse to single space
- **Formatting noise**: Build system output and caret lines stripped
- **Line number references in text**: `1 |` → `N |`, `| ^` → `| ^`

Only **error messages** are compared (warnings are extracted but not validated).

Example: actual build output
```
/absolute/path/file.cpp:42:3: error: static assertion failed: message
42 | code here
   | ^        ~~~
```

Gets normalized to:
```
file.cpp: error: static assertion failed: message
```

And compared against the normalized baseline. This ensures tests catch *semantic* changes without false positives from line number shifts or absolute paths.

## Typical Workflow Example

### Add a New Test

```bash
# 1. Write failing test
cat > tests/static_fail/test_negative_inversion.cpp << 'EOF'
#include "test_simple_include.hpp"

struct NonInvertible {};

int main() {
    // This should fail at compile time
    static_assert(has_inverse_v<NonInvertible>);
}
EOF

# 2. Register in CMakeLists.txt
compile_check(reflection_ad "static_fail/test_negative_inversion.cpp")

# 3. Create baseline with Clang
cmake --build build/cmake --target rebase-reflection_ad.static_fail.test_negative_inversion

# 4. Review and commit
git add tests/static_fail/test_negative_inversion.fail.txt

# 5. Verify with GCC
# (switch to GCC CMake session)
cmake --build build/cmake --target rebase-reflection_ad.static_fail.test_negative_inversion
git add tests/static_fail/test_negative_inversion.fail.txt

# 6. Run tests
ctest --test-dir build/cmake -R test_negative_inversion
```

### Update an Intentional Change

If you intentionally change code that affects error messages:

```bash
# 1. Make the code change
# ... edit code ...

# 2. Rebase with current compiler
cmake --build build/cmake --target rebase-reflection_ad.static_fail.inverse_of_non_invertible

# 3. Review the diff
git diff tests/static_fail/inverse_of_non_invertible.fail.txt

# 4. If correct, commit
git add tests/static_fail/inverse_of_non_invertible.fail.txt

# 5. Repeat for other compilers if needed
cmake --build build/cmake --target rebase-reflection_ad.static_fail.inverse_of_non_invertible
```

## Debugging Test Failures

If a test fails, the output shows the mismatch:

```
Expected (normalized):
[file.cpp: error: static assertion failed: ...]

Actual (normalized):
[file.cpp: error: different message]
```

**Common causes:**
- Intentional code change → rebase the baseline
- Compiler update → rebase with new compiler version
- Typo in error message → fix the code
- Missing error → add the `static_assert` or diagnostic

**To fix:** Run the rebase command, review changes with `git diff`, and commit if correct.

## CMake API

Register failing compilation tests in `CMakeLists.txt`:

```cmake
compile_check(group_name "relative/path/to/test.cpp")
```

This generates:
- **Test**: `${group_name}.${test_name}` (run with ctest)
- **Rebase target**: `rebase-${group_name}.${test_name}` (update baseline)

## Reference

- **Test macros**: See `test_simple_include.hpp` for `EXPECT_EQUAL`, `EXPECT_NEAR_REL`, etc.
- **Validation script**: `validate_compile_output.cmake` (extracts compiler section, normalizes, compares)
- **Extraction script**: `extract_error_output.cmake` (creates/updates baseline)
