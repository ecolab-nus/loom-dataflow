# Common utilities for LOOM passes
# Provides reusable function cloning and enumeration utilities

add_library(LoomPassesCommon
  src/utils.cpp
  src/block_loading_pattern.cpp
  src/hardware_info.cpp
)

target_include_directories(LoomPassesCommon
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    # Include LCS headers for constraint space utilities
    ${CMAKE_SOURCE_DIR}/lib/lcs/inc
    # Include the generated Loom dialect headers
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
    # Include the generated Dataflow dialect headers
    ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
    # Include affine directory for tiling and parallel-to-for
    ${CMAKE_SOURCE_DIR}/lib/passes/affine/inc
)

target_link_libraries(LoomPassesCommon
  PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRSupport
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    loomConstraintSystem
    loomDialect
    loomDataflowDialect
)

# Depend on the generated Loom dialect headers
add_dependencies(LoomPassesCommon
    LOOMLoomOpsIncGen
    LOOMLoomDialectIncGen
)
