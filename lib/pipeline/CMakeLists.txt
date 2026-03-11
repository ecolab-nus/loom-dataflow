# Loom pipeline shared library
# Exposes the full Materialize → Canonicalize → BridgeToOSB →
# LowerAffineWithAttr → One-Shot Bufferization → Canonicalize → CSE
# pipeline as a C API callable from Python via ctypes.

add_library(loom_pipeline SHARED
    ${CMAKE_CURRENT_LIST_DIR}/loom_pipeline_api.cpp
)

target_include_directories(loom_pipeline PRIVATE
    # Pass headers (inc/ + generated)
    ${CMAKE_SOURCE_DIR}/lib/passes/loom-opt/inc
    ${CMAKE_SOURCE_DIR}/lib/passes/common/inc
    ${CMAKE_BINARY_DIR}/lib/passes/loom-opt
    # Dialect generated headers
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
    ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
    # Dialect source headers (for Transforms/BufferizableOpInterfaceImpl.h)
    ${CMAKE_SOURCE_DIR}/lib/loom-dialect/Transforms
    # MLIR system includes
    ${MLIR_INCLUDE_DIRS}
)

target_compile_options(loom_pipeline PRIVATE -fno-rtti)

set_target_properties(loom_pipeline PROPERTIES
    # Only export the two C API symbols; hide all C++ internals
    C_VISIBILITY_PRESET   hidden
    CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Union of dependencies from canonicalize + one_shot_bufferize executables
target_link_libraries(loom_pipeline PRIVATE
    loomTritonSharedPasses
    loomDialectTransforms
    loomDataflowDialect
    loomDialect
    LoomPassesCommon
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

mlir_check_all_link_libraries(loom_pipeline)
