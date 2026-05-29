# Common utilities for LOOM passes
# Provides reusable function cloning and enumeration utilities

add_library(LoomPassesCommon
  src/utils.cpp
  src/trace_shape.cpp
  src/block_loading_pattern.cpp
  src/hardware_discovery.cpp
  src/hardware_mapping.cpp
  src/affine_utils.cpp
  src/index_mapping.cpp
  src/driver_utils.cpp
  src/ssa_utils.cpp
  src/loop_iv_dependency.cpp
  src/binary_scalar_chain.cpp
)

target_include_directories(LoomPassesCommon
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    # Include LCS headers for constraint space utilities
    ${CMAKE_SOURCE_DIR}/lib/lcs/inc
    # Include the generated Loom dialect headers
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
    # Include the generated ADL dialect headers
    ${CMAKE_BINARY_DIR}/lib/adl-dialect/IR
    # Include the source directory for other Loom internal headers
    ${PROJECT_SOURCE_DIR}/lib/passes/loom-opt/inc
)

target_link_libraries(LoomPassesCommon
  PUBLIC
    LoomAnalysis
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRAsmParser
    MLIRBytecodeReader
    MLIRFuncDialect
    MLIRSupport
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRTensorDialect
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTensorTilingInterfaceImpl
    MLIRLinalgDialect
    MLIRControlFlowDialect
    loomDialect
    loomADLDialect
)

# Depend on the generated Loom dialect headers
add_dependencies(LoomPassesCommon
    LOOMLoomOpsIncGen
    LOOMLoomDialectIncGen
)
