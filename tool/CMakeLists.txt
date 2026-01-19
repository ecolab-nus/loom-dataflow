# Root tools for affine and loom-opt

# Common include dirs for all tools
set(LOOM_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/loom-opt/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/constraint-opt/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

# Affine tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/affine)

# Triton-shared tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-opt/single_stage)

# Loom-constraint tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-constraint/single_stage)

# Combined pipeline tool
add_executable(ttshared-opt
  ${CMAKE_CURRENT_LIST_DIR}/loom-opt/ttshared-opt.cpp
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
  loomDialect
)
mlir_check_all_link_libraries(ttshared-opt)

target_compile_options(ttshared-opt PRIVATE -fno-rtti)


# Dataflow-dialect tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataflow-dialect)

# Resource-system demos
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/resource-system)

# Custom LSP server
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-lsp-server)
