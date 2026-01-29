# Root tools for affine and triton-shared

# Common include dirs for all tools
set(LOOM_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
)

# Triton-shared tools

# Triton-shared tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/triton-shared/single_stage)

# Combined pipeline tool
add_executable(ttshared-opt
  ${CMAKE_CURRENT_LIST_DIR}/triton-shared/ttshared-opt.cpp
)

target_include_directories(ttshared-opt PRIVATE ${MLIR_INCLUDE_DIRS} ${LOOM_PASSES_INC})

target_link_libraries(ttshared-opt PRIVATE
  loomTritonSharedPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRRegisterAllDialects
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
)
mlir_check_all_link_libraries(ttshared-opt)

target_compile_options(ttshared-opt PRIVATE -fno-rtti)


# Dataflow-dialect tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataflow-dialect)

# Resource-system demos
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/resource-system)
