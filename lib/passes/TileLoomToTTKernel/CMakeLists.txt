# TileLoomToTTKernel pass library
# Provides conversion patterns from tiled Loom IR to TTKernel dialect

add_library(loomTileLoomToTTKernel STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/MemoryOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ComputeOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/FuncOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/TileLoomToTTKernel.cpp
)

target_include_directories(loomTileLoomToTTKernel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Link MLIR components
target_link_libraries(loomTileLoomToTTKernel PUBLIC
    MLIRIR
    MLIRPass
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
    LLVMSupport
)
mlir_check_all_link_libraries(loomTileLoomToTTKernel)

target_compile_options(loomTileLoomToTTKernel PRIVATE -fno-rtti)
