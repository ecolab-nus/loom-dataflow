# Root tools for affine and triton-shared

# Common include dirs for all tools
set(TMD_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/common
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/triton-shared
  ${CMAKE_BINARY_DIR}/lib/scaleout/dataflow-dialect/IR
)

# Affine tools
add_executable(tmd_affine_explore
  ${CMAKE_CURRENT_LIST_DIR}/affine/affine_explore_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(tmd_affine_explore PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_affine_explore PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  tmdDataflowDialect
)
mlir_check_all_link_libraries(tmd_affine_explore)

target_compile_options(tmd_affine_explore PRIVATE -fno-rtti)

add_executable(tmd_affine_tile
  ${CMAKE_CURRENT_LIST_DIR}/affine/affine_tile_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_tile_pass.cpp
)

target_include_directories(tmd_affine_tile PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_affine_tile PRIVATE
  MLIRIR
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  LLVMSupport
)
mlir_check_all_link_libraries(tmd_affine_tile)

target_compile_options(tmd_affine_tile PRIVATE -fno-rtti)

add_executable(tmd_affine_analyze
  ${CMAKE_CURRENT_LIST_DIR}/affine/affine_analyze_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/input_sharing_analysis.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(tmd_affine_analyze PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_affine_analyze PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
  tmdDataflowDialect
)
mlir_check_all_link_libraries(tmd_affine_analyze)

# Triton-shared tools
add_executable(tmd_triton_shared_explore
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_explore_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(tmd_triton_shared_explore PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_triton_shared_explore PRIVATE
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
mlir_check_all_link_libraries(tmd_triton_shared_explore)

target_compile_options(tmd_triton_shared_explore PRIVATE -fno-rtti)

add_executable(tmd_triton_shared_affinize
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_affinize_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/triton-shared/triton_shared_affinize.cpp
)

target_include_directories(tmd_triton_shared_affinize PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_triton_shared_affinize PRIVATE
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
mlir_check_all_link_libraries(tmd_triton_shared_affinize)

target_compile_options(tmd_triton_shared_affinize PRIVATE -fno-rtti)

add_executable(tmd_triton_shared_grid_to_parallel
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_grid_to_parallel_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/triton-shared/triton_shared_grid_to_parallel.cpp
)

target_include_directories(tmd_triton_shared_grid_to_parallel PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_triton_shared_grid_to_parallel PRIVATE
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
mlir_check_all_link_libraries(tmd_triton_shared_grid_to_parallel)

target_compile_options(tmd_triton_shared_grid_to_parallel PRIVATE -fno-rtti)

# Combined pipeline tool
add_executable(tmd_triton_shared_to_affine
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_to_affine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/triton-shared/triton_shared_affinize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/triton-shared/triton_shared_grid_to_parallel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_parallel_to_for.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/scaleout/passes/affine/affine_tile.cpp
)

target_include_directories(tmd_triton_shared_to_affine PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(tmd_triton_shared_to_affine PRIVATE
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
mlir_check_all_link_libraries(tmd_triton_shared_to_affine)

target_compile_options(tmd_triton_shared_to_affine PRIVATE -fno-rtti)
