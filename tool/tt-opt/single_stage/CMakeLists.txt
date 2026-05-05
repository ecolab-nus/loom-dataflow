# TT-opt single-stage tools

add_executable(fold_zero_fill_linalg
  ${CMAKE_CURRENT_LIST_DIR}/fold_zero_fill_linalg_main.cpp
)

target_include_directories(fold_zero_fill_linalg PRIVATE
  ${MLIR_INCLUDE_DIRS} 
  ${LOOM_PASSES_INC}
  ${PROJECT_SOURCE_DIR}/lib/passes/tt-opt/inc
)

target_link_libraries(fold_zero_fill_linalg PRIVATE
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
  loomADLDialect
  loomDialect
)
mlir_check_all_link_libraries(fold_zero_fill_linalg)

target_compile_options(fold_zero_fill_linalg PRIVATE -fno-rtti)

set_target_properties(fold_zero_fill_linalg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
