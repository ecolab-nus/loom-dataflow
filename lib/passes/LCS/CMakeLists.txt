# LCS (Loom Compute Schedule) analysis library

add_library(loomLCS STATIC
    src/constraint_expr.cpp
    src/expr.cpp
    src/staged_etg_builder.cpp
    src/lcs_utils.cpp
)

target_include_directories(loomLCS PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(loomLCS PUBLIC
    LoomPassesCommon
    MLIRIR
    MLIRPass
    MLIRSupport
    LLVMSupport
    MLIRAffineDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRLinalgDialect
    loomDialect
)

add_dependencies(loomLCS LOOMLoomOpsIncGen LOOMLoomDialectIncGen)

target_compile_options(loomLCS PRIVATE -fno-rtti)
