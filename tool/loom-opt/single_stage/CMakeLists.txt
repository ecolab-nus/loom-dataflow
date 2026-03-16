# Single-stage Triton-shared tools

# enumerate_hw_mapping
add_executable(enumerate_hw_mapping
  ${CMAKE_CURRENT_LIST_DIR}/enumerate_hw_mapping_main.cpp
)

target_include_directories(enumerate_hw_mapping PRIVATE ${MLIR_INCLUDE_DIRS} ${LOOM_PASSES_INC})

target_link_libraries(enumerate_hw_mapping PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(enumerate_hw_mapping)

target_compile_options(enumerate_hw_mapping PRIVATE -fno-rtti)

set_target_properties(enumerate_hw_mapping PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)



add_executable(analyze_reuse
  ${CMAKE_CURRENT_LIST_DIR}/analyze_reuse_main.cpp
)

target_include_directories(analyze_reuse PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(analyze_reuse PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(analyze_reuse)

target_compile_options(analyze_reuse PRIVATE -fno-rtti)

set_target_properties(analyze_reuse PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# enumerate_copy_broadcast: Enumerate interconnect broadcast choices for copy operations
add_executable(enumerate_copy_broadcast
  ${CMAKE_CURRENT_LIST_DIR}/enumerate_copy_broadcast_main.cpp
)

target_include_directories(enumerate_copy_broadcast PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(enumerate_copy_broadcast PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(enumerate_copy_broadcast)

target_compile_options(enumerate_copy_broadcast PRIVATE -fno-rtti)

set_target_properties(enumerate_copy_broadcast PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# hoist_block_loading: Hoist block loading operations
add_executable(hoist_block_loading
  ${CMAKE_CURRENT_LIST_DIR}/hoist_block_loading_main.cpp
)

target_include_directories(hoist_block_loading PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(hoist_block_loading PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(hoist_block_loading)

target_compile_options(hoist_block_loading PRIVATE -fno-rtti)

set_target_properties(hoist_block_loading PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# canonicalize: Materialize and canonicalize IR
add_executable(canonicalize
  ${CMAKE_CURRENT_LIST_DIR}/canonicalize_main.cpp
)

target_include_directories(canonicalize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(canonicalize PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(canonicalize)

target_compile_options(canonicalize PRIVATE -fno-rtti)

set_target_properties(canonicalize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# memory_binding: Bind memory allocations to tensor operations
add_executable(memory_binding
  ${CMAKE_CURRENT_LIST_DIR}/memory_binding_main.cpp
)

target_include_directories(memory_binding PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(memory_binding PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(memory_binding)

target_compile_options(memory_binding PRIVATE -fno-rtti)

set_target_properties(memory_binding PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# tensor_canonicalize: Specialize LinalgOp destinations and fold redundant tensors
add_executable(tensor_canonicalize
  ${CMAKE_CURRENT_LIST_DIR}/tensor_canonicalize_main.cpp
)

target_include_directories(tensor_canonicalize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(tensor_canonicalize PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRControlFlowDialect
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(tensor_canonicalize)

target_compile_options(tensor_canonicalize PRIVATE -fno-rtti)

set_target_properties(tensor_canonicalize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# one_shot_bufferize: One-Shot Bufferization for LOOM
add_executable(one_shot_bufferize
  ${CMAKE_CURRENT_LIST_DIR}/one_shot_bufferize_main.cpp
)

target_include_directories(one_shot_bufferize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
  ${PROJECT_SOURCE_DIR}/lib/loom-dialect/Transforms
  ${PROJECT_SOURCE_DIR}/build/lib/dataflow-dialect/IR
)

target_link_libraries(one_shot_bufferize PRIVATE
  loomTritonSharedPasses
  loomDialectTransforms
  loomDataflowDialect
  loomDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRAffineToStandard
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRTensorTransforms
  MLIRAsmParser
  MLIRBytecodeReader
)
mlir_check_all_link_libraries(one_shot_bufferize)

target_compile_options(one_shot_bufferize PRIVATE -fno-rtti)

set_target_properties(one_shot_bufferize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# static_memory_analyser: Symbolic shape inference and bucketing
add_executable(static_memory_analyser
  ${CMAKE_CURRENT_LIST_DIR}/static_memory_analyser_main.cpp
)

target_include_directories(static_memory_analyser PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(static_memory_analyser PRIVATE
  LoomPassesCommon
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(static_memory_analyser)

target_compile_options(static_memory_analyser PRIVATE -fno-rtti)

set_target_properties(static_memory_analyser PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# staged_etg: Staged Execution Task Graph analysis
add_executable(staged_etg
  ${CMAKE_CURRENT_LIST_DIR}/staged_etg_main.cpp
)

target_include_directories(staged_etg PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(staged_etg PRIVATE
  loomLCS
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
  MLIRMathDialect
)
mlir_check_all_link_libraries(staged_etg)

target_compile_options(staged_etg PRIVATE -fno-rtti)

set_target_properties(staged_etg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# parse_compute_ops: Parse and dump compute MLIR files from a directory
add_executable(parse_compute_ops
  ${CMAKE_CURRENT_LIST_DIR}/parse_compute_ops_main.cpp
)

target_include_directories(parse_compute_ops PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(parse_compute_ops PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRAsmParser
  MLIRBytecodeReader
  loomDialect
)
mlir_check_all_link_libraries(parse_compute_ops)

target_compile_options(parse_compute_ops PRIVATE -fno-rtti)

set_target_properties(parse_compute_ops PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
