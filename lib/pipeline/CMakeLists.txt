# Loom pipeline libraries and Python bindings
#
# Builds:
#   1. loom_pipeline_core  - static library with all pipeline implementations
#   2. _loom_pipeline      - pybind11 Python extension module

# Common include directories for pipeline sources
set(LOOM_PIPELINE_INCLUDES
    # Pipeline headers (inc/ + generated version header)
    ${CMAKE_CURRENT_LIST_DIR}/inc
    ${CMAKE_BINARY_DIR}/lib/pipeline
    # Pass headers (inc/ + generated)
    ${CMAKE_SOURCE_DIR}/lib/passes/loom-opt/inc
    ${CMAKE_SOURCE_DIR}/lib/passes
    ${CMAKE_SOURCE_DIR}/lib/passes/common/inc
    ${CMAKE_SOURCE_DIR}/lib/passes/lcs/inc
    ${CMAKE_BINARY_DIR}/lib/passes/loom-opt
    ${CMAKE_BINARY_DIR}/lib/passes/tt-opt
    # Dialect generated headers
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
    # Dialect source headers (for Transforms/BufferizableOpInterfaceImpl.h)
    ${CMAKE_SOURCE_DIR}/lib/loom-dialect/Transforms
    # MLIR system includes
    ${MLIR_INCLUDE_DIRS}
)

# Union of all MLIR/Loom link libraries needed by both pipelines
set(LOOM_PIPELINE_LIBS
    loomTritonSharedPasses
    loomTTOptPasses
    loomDialectTransforms
    loomDialect
    LoomPassesCommon
    loomLCS
    # Core MLIR
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    # Dialect libraries
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRControlFlowDialect
    MLIRMathDialect
    # Conversion + transforms
    MLIRAffineToStandard
    MLIRLinalgTransforms
    MLIRTensorTransforms
    # Reader/Writer
    MLIRAsmParser
    MLIRBytecodeReader
    # LLVM
    LLVMSupport
)

# ---------------------------------------------------------------------------
# 1. Static library: loom_pipeline_core
#    Contains both exploration and materialization pipeline implementations.
# ---------------------------------------------------------------------------
add_library(loom_pipeline_core STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/loom_materialization_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/loom_exploration_pipeline.cpp
)

target_include_directories(loom_pipeline_core PRIVATE ${LOOM_PIPELINE_INCLUDES})
target_compile_options(loom_pipeline_core PRIVATE -fno-rtti)
target_link_libraries(loom_pipeline_core PRIVATE ${LOOM_PIPELINE_LIBS})

# ---------------------------------------------------------------------------
# 2. pybind11 module: _loom_pipeline
#    Python extension exposing both pipelines + version checking.
# ---------------------------------------------------------------------------
pybind11_add_module(_loom_pipeline
    ${CMAKE_CURRENT_LIST_DIR}/src/loom_pybind.cpp
)

target_include_directories(_loom_pipeline PRIVATE ${LOOM_PIPELINE_INCLUDES})
# NOTE: The pybind11 binding file is compiled WITH RTTI (default).
# pybind11's descr::types() uses typeid() unconditionally, which requires RTTI.
# This is safe because loom_pybind.cpp only calls pipeline functions via their
# std::string-based C++ API and does not interact with MLIR types directly.
# The actual MLIR pipeline code in loom_pipeline_core is still compiled with
# -fno-rtti as required by MLIR.
target_link_libraries(_loom_pipeline PRIVATE loom_pipeline_core)

set_target_properties(_loom_pipeline PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ---------------------------------------------------------------------------
# Install rules for Python packaging (scikit-build-core)
# ---------------------------------------------------------------------------
install(TARGETS _loom_pipeline LIBRARY DESTINATION loom_pipeline)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/__init__.py DESTINATION loom_pipeline)
