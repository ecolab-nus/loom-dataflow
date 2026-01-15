# Single-stage constraint optimization tools

# constraint_canonicalize
add_executable(constraint_canonicalize
  ${CMAKE_CURRENT_LIST_DIR}/constraint_canonicalize_main.cpp
)

target_include_directories(constraint_canonicalize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/lib/passes/constraint-opt/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

target_link_libraries(constraint_canonicalize PRIVATE
  loomConstraintOptPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(constraint_canonicalize)

target_compile_options(constraint_canonicalize PRIVATE -fno-rtti)

set_target_properties(constraint_canonicalize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# constraint_decompose
add_executable(constraint_decompose
  ${CMAKE_CURRENT_LIST_DIR}/constraint_decompose_main.cpp
)

target_include_directories(constraint_decompose PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/lib/passes/constraint-opt/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

target_link_libraries(constraint_decompose PRIVATE
  loomConstraintOptPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(constraint_decompose)

target_compile_options(constraint_decompose PRIVATE -fno-rtti)

set_target_properties(constraint_decompose PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# constraint_factorize
add_executable(constraint_factorize
  ${CMAKE_CURRENT_LIST_DIR}/constraint_factorize_main.cpp
)

target_include_directories(constraint_factorize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/lib/passes/constraint-opt/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

target_link_libraries(constraint_factorize PRIVATE
  loomConstraintOptPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(constraint_factorize)

target_compile_options(constraint_factorize PRIVATE -fno-rtti)

set_target_properties(constraint_factorize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# constraint_linearize
add_executable(constraint_linearize
  ${CMAKE_CURRENT_LIST_DIR}/constraint_linearize_main.cpp
)

target_include_directories(constraint_linearize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/lib/passes/constraint-opt/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
  ${CMAKE_SOURCE_DIR}/lib/lcs/inc
)

target_link_libraries(constraint_linearize PRIVATE
  loomConstraintOptPasses
  loomConstraintSystem
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRAsmParser
  MLIRBytecodeReader
  loomDataflowDialect
  loomDialect
)
mlir_check_all_link_libraries(constraint_linearize)

target_compile_options(constraint_linearize PRIVATE -fno-rtti)

set_target_properties(constraint_linearize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
