# Root tools for affine and triton-shared

# Common include dirs for all tools
set(TMD_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
)

# Affine tools
add_executable(affine_explore
  ${CMAKE_CURRENT_LIST_DIR}/affine/affine_explore_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(affine_explore PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(affine_explore PRIVATE
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
mlir_check_all_link_libraries(affine_explore)

target_compile_options(affine_explore PRIVATE -fno-rtti)

add_executable(affine_tile
  ${CMAKE_CURRENT_LIST_DIR}/affine/affine_tile_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile_pass.cpp
)

target_include_directories(affine_tile PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(affine_tile PRIVATE
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
mlir_check_all_link_libraries(affine_tile)

target_compile_options(affine_tile PRIVATE -fno-rtti)

add_executable(affine_analyze
  ${CMAKE_CURRENT_LIST_DIR}/affine/affine_analyze_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/input_sharing_analysis.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(affine_analyze PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(affine_analyze PRIVATE
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
mlir_check_all_link_libraries(affine_analyze)

# Triton-shared tools
add_executable(triton_shared_explore
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_explore_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_parallel_to_for.cpp
)

target_include_directories(triton_shared_explore PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(triton_shared_explore PRIVATE
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
mlir_check_all_link_libraries(triton_shared_explore)

target_compile_options(triton_shared_explore PRIVATE -fno-rtti)

add_executable(triton_shared_affinize
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_affinize_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_affinize.cpp
)

target_include_directories(triton_shared_affinize PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(triton_shared_affinize PRIVATE
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
mlir_check_all_link_libraries(triton_shared_affinize)

target_compile_options(triton_shared_affinize PRIVATE -fno-rtti)

add_executable(triton_shared_grid_to_parallel
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_grid_to_parallel_main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_grid_to_parallel.cpp
)

target_include_directories(triton_shared_grid_to_parallel PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(triton_shared_grid_to_parallel PRIVATE
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
mlir_check_all_link_libraries(triton_shared_grid_to_parallel)

target_compile_options(triton_shared_grid_to_parallel PRIVATE -fno-rtti)

# Combined pipeline tool
add_executable(triton_shared_to_affine
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/triton_shared_to_affine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_affinize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/triton_shared_grid_to_parallel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/spatial_mapping.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_parallel_to_for.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/affine_tile.cpp
)

target_include_directories(triton_shared_to_affine PRIVATE ${MLIR_INCLUDE_DIRS} ${TMD_PASSES_INC})

target_link_libraries(triton_shared_to_affine PRIVATE
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
mlir_check_all_link_libraries(triton_shared_to_affine)

target_compile_options(triton_shared_to_affine PRIVATE -fno-rtti)

# Dataflow-dialect tool(s)
add_executable(df_parse_print
  ${CMAKE_CURRENT_LIST_DIR}/dataflow-dialect/df_parse_print.cpp
)

target_include_directories(df_parse_print PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${TMD_PASSES_INC}
)

target_link_libraries(df_parse_print PRIVATE
  tmdDataflowDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
)
mlir_check_all_link_libraries(df_parse_print)

# Resource-system demos
add_executable(resource_module_demo
  ${CMAKE_CURRENT_LIST_DIR}/resource-system/module_demo.cpp
)

target_link_libraries(resource_module_demo PRIVATE
  scaleout
)

target_include_directories(resource_module_demo PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../lib
)

add_executable(resource_demo
  ${CMAKE_CURRENT_LIST_DIR}/resource-system/resource_demo.cpp
)

target_link_libraries(resource_demo PRIVATE
  scaleout
)

target_include_directories(resource_demo PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../lib
)
