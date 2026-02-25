# Cost-model passes library
# Provides workload extraction + performance estimation and top-k pruning

set(LLVM_TARGET_DEFINITIONS inc/Passes.td)
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name LoomCostModel ${TBLGEN_INCLUDES})
add_public_tablegen_target(LoomCostModelPassesIncGen)

add_library(loomCostModelPasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/cost_model_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/workload_builder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/performance_model.cpp
)

target_include_directories(loomCostModelPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

target_include_directories(loomCostModelPasses PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(loomCostModelPasses PUBLIC
    LoomPassesCommon
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    LLVMSupport
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRBufferizationDialect
    MLIRAsmParser
    MLIRBytecodeReader
    loomDataflowDialect
    loomDialect
)
mlir_check_all_link_libraries(loomCostModelPasses)

add_dependencies(loomCostModelPasses LoomCostModelPassesIncGen)

target_compile_options(loomCostModelPasses PRIVATE -fno-rtti)
