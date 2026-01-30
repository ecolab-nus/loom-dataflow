# Single-stage constraint optimization tools

# function to create single stage tool
function(add_single_stage_tool name)
  add_executable(${name}
    ${CMAKE_CURRENT_LIST_DIR}/${name}_main.cpp
  )

  target_include_directories(${name} PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/lib/passes/constraint-opt/inc
    ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
    ${CMAKE_SOURCE_DIR}/lib/lcs/inc
  )

  target_link_libraries(${name} PRIVATE
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
  mlir_check_all_link_libraries(${name})

  target_compile_options(${name} PRIVATE -fno-rtti)

  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction()

add_single_stage_tool(constraint_canonicalize)
add_single_stage_tool(constraint_decompose)
add_single_stage_tool(constraint_factorize)
add_single_stage_tool(constraint_linearize)
add_single_stage_tool(compress_intermediate_var)
add_single_stage_tool(constraint_simplify)
