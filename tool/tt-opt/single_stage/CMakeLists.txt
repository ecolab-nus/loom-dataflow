# TT-opt single-stage tools

add_executable(fuse_fill_matmul
  ${CMAKE_CURRENT_LIST_DIR}/fuse_fill_matmul_main.cpp
)

target_include_directories(fuse_fill_matmul PRIVATE 
  ${MLIR_INCLUDE_DIRS} 
  ${LOOM_PASSES_INC}
  ${PROJECT_SOURCE_DIR}/lib/passes/tt-opt/inc
)

target_link_libraries(fuse_fill_matmul PRIVATE
  loomTTOptPasses
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
mlir_check_all_link_libraries(fuse_fill_matmul)

target_compile_options(fuse_fill_matmul PRIVATE -fno-rtti)

set_target_properties(fuse_fill_matmul PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
