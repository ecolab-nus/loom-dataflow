# Affine tools

add_executable(affine_explore
  ${CMAKE_CURRENT_LIST_DIR}/affine_explore_main.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/common/spatial_mapping.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_tile.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_parallel_to_for.cpp
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

set_target_properties(affine_explore PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(affine_tile
  ${CMAKE_CURRENT_LIST_DIR}/affine_tile_main.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_tile.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_tile_pass.cpp
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

set_target_properties(affine_tile PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(affine_analyze
  ${CMAKE_CURRENT_LIST_DIR}/affine_analyze_main.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/common/spatial_mapping.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/common/input_sharing_analysis.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_tile.cpp
  ${CMAKE_SOURCE_DIR}/lib/passes/affine/affine_parallel_to_for.cpp
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

set_target_properties(affine_analyze PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


