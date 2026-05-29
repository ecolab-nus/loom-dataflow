add_executable(loom-opt
  loom-opt.cpp
)

target_include_directories(loom-opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
  ${CMAKE_BINARY_DIR}/lib/adl-dialect/IR
)

target_link_libraries(loom-opt PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
  loomADLDialect
  loomDialect
)

mlir_check_all_link_libraries(loom-opt)

target_compile_options(loom-opt PRIVATE -fno-rtti)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/single_stage)
