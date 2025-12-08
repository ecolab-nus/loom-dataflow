# Dataflow-dialect tool(s)

add_executable(df_example_parse
  ${CMAKE_CURRENT_LIST_DIR}/df_example_parse.cpp
)

target_include_directories(df_example_parse PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
)

target_link_libraries(df_example_parse PRIVATE
  loomDataflowDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
)
mlir_check_all_link_libraries(df_example_parse)

set_target_properties(df_example_parse PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


