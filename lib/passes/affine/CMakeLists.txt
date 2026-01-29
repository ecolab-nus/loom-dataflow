# Affine passes library
# Provides affine transformation passes for MLIR

add_library(loomAffinePasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/affine_parallel_to_for.cpp
)

target_include_directories(loomAffinePasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Link MLIR components
target_link_libraries(loomAffinePasses PUBLIC
    MLIRIR
    MLIRPass
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRSupport
    LLVMSupport
)
mlir_check_all_link_libraries(loomAffinePasses)

target_compile_options(loomAffinePasses PRIVATE -fno-rtti)
