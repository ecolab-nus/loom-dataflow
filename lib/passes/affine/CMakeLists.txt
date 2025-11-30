# Affine passes library
# Provides affine transformation passes for MLIR

add_library(tmdAffinePasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/affine_parallel_to_for.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/affine_tile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/affine_tile_pass.cpp
)

target_include_directories(tmdAffinePasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Link MLIR components
target_link_libraries(tmdAffinePasses PUBLIC
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
mlir_check_all_link_libraries(tmdAffinePasses)

target_compile_options(tmdAffinePasses PRIVATE -fno-rtti)
