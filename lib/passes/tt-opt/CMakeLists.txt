# TT-Opt passes library

set(LLVM_TARGET_DEFINITIONS inc/Passes.td)
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name LoomTTOpt ${TBLGEN_INCLUDES})
add_public_tablegen_target(LoomTTOptPassesIncGen)

add_library(loomTTOptPasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/convert_zero_fill_linalg_matmul_to_loom_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/fold_zero_fill_linalg_pass.cpp
)

target_include_directories(loomTTOptPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/lib/passes/loom-opt/inc
)

target_link_libraries(loomTTOptPasses PUBLIC
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    LLVMSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    loomDialect
)
mlir_check_all_link_libraries(loomTTOptPasses)
add_dependencies(loomTTOptPasses LoomTTOptPassesIncGen)

target_compile_options(loomTTOptPasses PRIVATE -fno-rtti)
