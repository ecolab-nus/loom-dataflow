# Static memory analysis library.
# Provides the MemoryAnalysisContext API consumed by memory_binding_pass and
# the standalone static_memory_analyser CLI tool.

add_library(LoomAnalysis
  src/static_memory_analyser.cpp
)

target_include_directories(LoomAnalysis
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    # utils.h (SymbolicDim) lives in LoomPassesCommon's include tree.
    # Expose it here so downstream consumers of LoomAnalysis can find it via
    # the single include of static_memory_analyser.h.
    ${CMAKE_SOURCE_DIR}/lib/passes/common/inc
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
    ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
)

target_link_libraries(LoomAnalysis
  PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRSupport
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    loomDialect
    loomDataflowDialect
)

add_dependencies(LoomAnalysis
  LOOMLoomOpsIncGen
  LOOMLoomDialectIncGen
)
