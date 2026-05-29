# ADL-dialect tool(s)

add_executable(adl_parse
  ${CMAKE_CURRENT_LIST_DIR}/adl_parse.cpp
)

target_include_directories(adl_parse PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/lib/adl-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

target_link_libraries(adl_parse PRIVATE
  loomADLDialect
  loomDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRMathDialect
)
mlir_check_all_link_libraries(adl_parse)

set_target_properties(adl_parse PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
