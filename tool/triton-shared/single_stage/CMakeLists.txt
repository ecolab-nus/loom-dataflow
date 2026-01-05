# Single-stage Triton-shared tools

# explore_mapping
add_executable(explore_mapping
  ${CMAKE_CURRENT_LIST_DIR}/explore_mapping_main.cpp
)

target_include_directories(explore_mapping PRIVATE ${MLIR_INCLUDE_DIRS} ${LOOM_PASSES_INC})

target_link_libraries(explore_mapping PRIVATE
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
mlir_check_all_link_libraries(explore_mapping)

target_compile_options(explore_mapping PRIVATE -fno-rtti)

set_target_properties(explore_mapping PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(affinize
  ${CMAKE_CURRENT_LIST_DIR}/affinize_main.cpp
)

target_include_directories(affinize PRIVATE ${MLIR_INCLUDE_DIRS} ${LOOM_PASSES_INC})

target_link_libraries(affinize PRIVATE
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
mlir_check_all_link_libraries(affinize)

target_compile_options(affinize PRIVATE -fno-rtti)

set_target_properties(affinize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(grid_to_parallel
  ${CMAKE_CURRENT_LIST_DIR}/grid_to_parallel_main.cpp
)

target_include_directories(grid_to_parallel PRIVATE ${MLIR_INCLUDE_DIRS} ${LOOM_PASSES_INC})

target_link_libraries(grid_to_parallel PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(grid_to_parallel)

target_compile_options(grid_to_parallel PRIVATE -fno-rtti)

set_target_properties(grid_to_parallel PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(annotate_reuse
  ${CMAKE_CURRENT_LIST_DIR}/annotate_reuse_main.cpp
)

target_include_directories(annotate_reuse PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(annotate_reuse PRIVATE
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
mlir_check_all_link_libraries(annotate_reuse)

target_compile_options(annotate_reuse PRIVATE -fno-rtti)

set_target_properties(annotate_reuse PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# explore_alloc_copy_mapping: Annotate reuse and explore alloc/copy mappings
add_executable(explore_alloc_copy_mapping
  ${CMAKE_CURRENT_LIST_DIR}/explore_alloc_copy_mapping_main.cpp
)

target_include_directories(explore_alloc_copy_mapping PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(explore_alloc_copy_mapping PRIVATE
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
mlir_check_all_link_libraries(explore_alloc_copy_mapping)

target_compile_options(explore_alloc_copy_mapping PRIVATE -fno-rtti)

set_target_properties(explore_alloc_copy_mapping PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# tile_scf_for_to_l1: Tile scf.for loops to fit L1
add_executable(tile_scf_for_to_l1
  ${CMAKE_CURRENT_LIST_DIR}/tile_scf_for_to_l1_main.cpp
)

target_include_directories(tile_scf_for_to_l1 PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(tile_scf_for_to_l1 PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
)
mlir_check_all_link_libraries(tile_scf_for_to_l1)

target_compile_options(tile_scf_for_to_l1 PRIVATE -fno-rtti)

set_target_properties(tile_scf_for_to_l1 PROPERTIES
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


