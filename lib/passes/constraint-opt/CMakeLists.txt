# Constraint Optimization Passes Library
# Provides passes for polynomial constraint optimization

set(LLVM_TARGET_DEFINITIONS inc/Passes.td)
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name LoomConstraintOpt ${TBLGEN_INCLUDES})
add_public_tablegen_target(LoomConstraintOptPassesIncGen)

add_library(loomConstraintOptPasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_canonicalize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_factorize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_decompose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_linearize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/compress_intermediate_var.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_simplify.cpp
)

target_include_directories(loomConstraintOptPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(loomConstraintOptPasses PUBLIC
    LoomPassesCommon
    loomConstraintSystem
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    LLVMSupport
    loomDialect
)
add_dependencies(loomConstraintOptPasses LoomConstraintOptPassesIncGen)

mlir_check_all_link_libraries(loomConstraintOptPasses)

target_compile_options(loomConstraintOptPasses PRIVATE -fno-rtti)
