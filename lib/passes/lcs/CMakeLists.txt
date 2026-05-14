# LCS (Loom Compute Schedule) analysis library

add_library(loomLCS STATIC
    src/constraint_expr.cpp
    src/expr.cpp
    src/hw_alignment.cpp
    src/l1_footprint_estimator.cpp
    src/staged_etg_builder.cpp
    src/lcs_utils.cpp
    src/hw_op_registry.cpp
    src/hw_op_registry_compute.cpp
    src/hw_op_registry_datamover.cpp
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
    MLIRParser
    MLIRFuncDialect
    loomDialect
)

add_dependencies(loomLCS LOOMLoomOpsIncGen LOOMLoomDialectIncGen)

target_compile_options(loomLCS PRIVATE -fno-rtti)
