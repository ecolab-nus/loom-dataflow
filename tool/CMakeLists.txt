# Root tools for affine and triton-shared

# Common include dirs for all tools
set(TMD_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
)

# Affine tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/affine)

# Triton-shared tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/triton-shared/single_stage)

# Combined pipeline tool
add_executable(ttshared-opt
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/ttshared-opt.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_affinize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_grid_to_parallel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_spatial_mapping_pass.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/reinterpret_cast_reuse.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/explore_alloc_copy_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/tile_scf_for_to_l1.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/const_dedup_cleanup.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_parallel_to_for.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile.cpp
)

target_include_directories(ttshared-opt PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(ttshared-opt PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRRegisterAllDialects
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  tmdDataflowDialect
)
mlir_check_all_link_libraries(ttshared-opt)

target_compile_options(ttshared-opt PRIVATE -fno-rtti)


# Dataflow-dialect tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataflow-dialect)

# Resource-system demos
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/resource-system)
