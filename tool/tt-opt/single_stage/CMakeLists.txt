# TT-opt single-stage tools

add_executable(tt-opt
  ${CMAKE_CURRENT_LIST_DIR}/tt-opt.cpp
)

target_include_directories(tt-opt PRIVATE
  ${MLIR_INCLUDE_DIRS} 
  ${LOOM_PASSES_INC}
  ${PROJECT_SOURCE_DIR}/lib/passes/tt-opt/inc
)

target_link_libraries(tt-opt PRIVATE
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
mlir_check_all_link_libraries(tt-opt)

target_compile_options(tt-opt PRIVATE -fno-rtti)

set_target_properties(tt-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
