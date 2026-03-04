# LCS (Loom Compute Schedule) analysis library

add_library(loomLCS STATIC
    src/StagedETG.cpp
)

target_include_directories(loomLCS PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(loomLCS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    LLVMSupport
    MLIRAffineDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRLinalgDialect
)

target_compile_options(loomLCS PRIVATE -fno-rtti)
