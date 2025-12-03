# Triton-shared passes library
# Provides passes for Triton-shared optimization pipeline

add_library(tmdTritonSharedPasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/block_loading_pattern.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/const_dedup_cleanup.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/explore_alloc_copy_mapping.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hoist_block_loading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/input_sharing_analysis.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/reinterpret_cast_reuse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/spatial_mapping.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/tile_scf_for_to_l1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_affinize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_grid_to_parallel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_spatial_mapping_pass.cpp
)

target_include_directories(tmdTritonSharedPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Triton-shared passes depend on affine passes
target_link_libraries(tmdTritonSharedPasses PUBLIC
    tmdAffinePasses
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
    MLIRBufferizationTransforms
    MLIRAsmParser
    MLIRBytecodeReader
    tmdDataflowDialect
)
mlir_check_all_link_libraries(tmdTritonSharedPasses)

target_compile_options(tmdTritonSharedPasses PRIVATE -fno-rtti)
