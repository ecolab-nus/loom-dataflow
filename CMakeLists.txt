cmake_minimum_required(VERSION 3.20.0)

# Read version from the single-source-of-truth VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LOOM_VERSION_FROM_FILE)
string(STRIP "${LOOM_VERSION_FROM_FILE}" LOOM_VERSION_FROM_FILE)

# Project name and version - MLIR requires both CXX and C languages
project(loom VERSION ${LOOM_VERSION_FROM_FILE} LANGUAGES CXX C)

# MLIR Configuration
# Set MLIR paths - these can be overridden via command line
if(NOT DEFINED MLIR_DIR)
    set(MLIR_DIR "/opt/llvm-mlir/lib/cmake/mlir" CACHE PATH "Path to MLIR cmake files")
endif()

if(NOT DEFINED LLVM_USE_LINKER)
    set(LLVM_USE_LINKER "lld" CACHE STRING "Linker to use")
endif()

# Find MLIR package
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set LLVM/MLIR paths
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Add LLVM/MLIR cmake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include necessary LLVM/MLIR modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Configure Python support if needed
if(MLIR_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif()

message(STATUS "MLIR support enabled")

# Generate version header from CMake project version
configure_file(
    ${CMAKE_SOURCE_DIR}/lib/pipeline/inc/loom_version.h.in
    ${CMAKE_BINARY_DIR}/lib/pipeline/loom_version.h
    @ONLY
)

# pybind11 for Python bindings
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()
message(STATUS "pybind11 found: ${pybind11_VERSION}")

# Enable position-independent code globally so all static libraries can be
# linked into libloom_pipeline.so without relocation errors.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Libraries
add_subdirectory(lib)

# Root-level tools
add_subdirectory(tool)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")