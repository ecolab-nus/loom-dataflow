# Single-stage loom-opt tools

# Helper: define a standard loom single-stage tool with common deps.
# Extra per-tool link libraries are appended via ${ARGN}.
function(add_loom_single_stage_tool name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${MLIR_INCLUDE_DIRS} ${LOOM_PASSES_INC})
  target_link_libraries(${name} PRIVATE
    loomTritonSharedPasses
    loomDataflowDialect
    loomDialect
    MLIRIR
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
    MLIRPass
    MLIRAsmParser
    MLIRBytecodeReader
    ${ARGN}
  )
  mlir_check_all_link_libraries(${name})
  target_compile_options(${name} PRIVATE -fno-rtti)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# ---- Core pipeline tools ----

# Step 1: Specialize LinalgOp destinations and fold redundant tensors
add_loom_single_stage_tool(tensor_canonicalize
  ${CMAKE_CURRENT_LIST_DIR}/tensor_canonicalize_main.cpp
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRControlFlowDialect
)

# Step 2: Bind memory allocations to tensor operations
add_loom_single_stage_tool(memory_binding
  ${CMAKE_CURRENT_LIST_DIR}/memory_binding_main.cpp
)

# Step 3: Enumerate spatial mappings from df.spatial_dim to affine.parallel
add_loom_single_stage_tool(enumerate_hw_mapping
  ${CMAKE_CURRENT_LIST_DIR}/enumerate_hw_mapping_main.cpp
)

# Step 4 (optional): Hoist block loading operations from innermost loops
add_loom_single_stage_tool(hoist_block_loading
  ${CMAKE_CURRENT_LIST_DIR}/hoist_block_loading_main.cpp
)

# Step 5: Annotate loom.subview with reuse patterns
add_loom_single_stage_tool(analyze_reuse
  ${CMAKE_CURRENT_LIST_DIR}/analyze_reuse_main.cpp
)

# Step 6: Enumerate per-copy broadcast choices and annotate allocs
add_loom_single_stage_tool(enumerate_copy_broadcast
  ${CMAKE_CURRENT_LIST_DIR}/enumerate_copy_broadcast_main.cpp
)

# Step 7: Build Staged Execution Task Graph and emit JSON constraint model
add_loom_single_stage_tool(staged_etg
  ${CMAKE_CURRENT_LIST_DIR}/staged_etg_main.cpp
  loomLCS
  MLIRMathDialect
)

# Step 8: Materialize symbolic block sizes and canonicalize IR
add_loom_single_stage_tool(canonicalize
  ${CMAKE_CURRENT_LIST_DIR}/canonicalize_main.cpp
)

# Step 9: One-Shot Bufferization — tensor ops → memref ops
# Extra include dirs and link libraries required for full bufferization pipeline.
add_executable(one_shot_bufferize
  ${CMAKE_CURRENT_LIST_DIR}/one_shot_bufferize_main.cpp
)
target_include_directories(one_shot_bufferize PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
  ${PROJECT_SOURCE_DIR}/lib/loom-dialect/Transforms
  ${PROJECT_SOURCE_DIR}/build/lib/dataflow-dialect/IR
)
target_link_libraries(one_shot_bufferize PRIVATE
  loomTritonSharedPasses
  loomDialectTransforms
  loomDataflowDialect
  loomDialect
  MLIRIR
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
  MLIRAffineToStandard
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRTensorTransforms
  MLIRAsmParser
  MLIRBytecodeReader
)
mlir_check_all_link_libraries(one_shot_bufferize)
target_compile_options(one_shot_bufferize PRIVATE -fno-rtti)
set_target_properties(one_shot_bufferize PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- Utility / debug tools ----

# static_memory_analyser: symbolic shape inference and buffer coloring debug CLI
add_loom_single_stage_tool(static_memory_analyser
  ${CMAKE_CURRENT_LIST_DIR}/static_memory_analyser_main.cpp
  LoomAnalysis
)
