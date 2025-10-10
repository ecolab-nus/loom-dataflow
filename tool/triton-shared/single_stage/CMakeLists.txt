# Single-stage Triton-shared tools

# explore_mapping
add_executable(explore_mapping
  ${CMAKE_CURRENT_LIST_DIR}/explore_mapping.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/common/spatial_mapping.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/common/reinterpret_cast_reuse.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_tile.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(explore_mapping PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(explore_mapping PRIVATE
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
  tmdDataflowDialect
)
mlir_check_all_link_libraries(explore_mapping)

target_compile_options(explore_mapping PRIVATE -fno-rtti)

set_target_properties(explore_mapping PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(affinize
  ${CMAKE_CURRENT_LIST_DIR}/affinize.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/triton-shared/triton_shared_affinize.cpp
)

target_include_directories(affinize PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(affinize PRIVATE
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
  tmdDataflowDialect
)
mlir_check_all_link_libraries(affinize)

target_compile_options(affinize PRIVATE -fno-rtti)

set_target_properties(affinize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(grid_to_parallel
  ${CMAKE_CURRENT_LIST_DIR}/grid_to_parallel.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/triton-shared/triton_shared_grid_to_parallel.cpp
)

target_include_directories(grid_to_parallel PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(grid_to_parallel PRIVATE
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
)
mlir_check_all_link_libraries(grid_to_parallel)

target_compile_options(grid_to_parallel PRIVATE -fno-rtti)

set_target_properties(grid_to_parallel PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(annotate_reuse
  ${CMAKE_CURRENT_LIST_DIR}/annotate_reuse.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/common/reinterpret_cast_reuse.cpp
)

target_include_directories(annotate_reuse PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${TMD_PASSES_INC}
)

target_link_libraries(annotate_reuse PRIVATE
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
  tmdDataflowDialect
)
mlir_check_all_link_libraries(annotate_reuse)

target_compile_options(annotate_reuse PRIVATE -fno-rtti)

set_target_properties(annotate_reuse PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


