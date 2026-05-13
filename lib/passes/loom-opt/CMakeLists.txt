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
    ${CMAKE_CURRENT_LIST_DIR}/src/enumerate_copy_broadcast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hoist_block_loading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/materialize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/analyze_reuse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/triton_shared_spatial_mapping_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg_destination_specialization_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/fold_redundant_extract_slice_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sink_fill_ops_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg_guarded_elementwise_fusion_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/canonical_bufferization_to_loom_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/loop_handoff_proxy_copy_insertion_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/lower_linalg_copy_to_loom_copy_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/memory_binding_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/view_to_reinterpret_cast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/lower_affine_with_attr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handoff_sync_insertion_pass.cpp
)

target_include_directories(loomTritonSharedPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    # ADL dialect generated headers
    ${CMAKE_BINARY_DIR}/lib/adl-dialect/IR
)

# Triton-shared passes depend on affine passes and common utilities
target_link_libraries(loomTritonSharedPasses PUBLIC
    LoomPassesCommon
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
    loomADLDialect
    loomDialect
)
mlir_check_all_link_libraries(loomTritonSharedPasses)
add_dependencies(loomTritonSharedPasses LoomTritonSharedPassesIncGen)

target_compile_options(loomTritonSharedPasses PRIVATE -fno-rtti)
