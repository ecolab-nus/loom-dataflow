# Triton-shared passes library
# Provides passes for Triton-shared optimization pipeline

set(LLVM_TARGET_DEFINITIONS inc/Passes.td)
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name LoomTritonShared ${TBLGEN_INCLUDES})
add_public_tablegen_target(LoomTritonSharedPassesIncGen)

add_library(loomTritonSharedPasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/const_dedup_cleanup.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/enumerate_copy_broadcast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hoist_block_loading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/input_sharing_analysis.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/loom_to_memref.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/materialize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/analyze_reuse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/staticize_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/tile_scf_for_to_l1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_affinize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_grid_to_parallel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_spatial_mapping_pass.cpp
)

target_include_directories(loomTritonSharedPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Triton-shared passes depend on affine passes and common utilities
target_link_libraries(loomTritonSharedPasses PUBLIC
    LoomPassesCommon
    loomAffinePasses
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
    loomDataflowDialect
    loomDialect
)
mlir_check_all_link_libraries(loomTritonSharedPasses)
add_dependencies(loomTritonSharedPasses LoomTritonSharedPassesIncGen)

target_compile_options(loomTritonSharedPasses PRIVATE -fno-rtti)
