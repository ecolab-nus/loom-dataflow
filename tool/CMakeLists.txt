# Root tools for affine and loom-opt

# Common include dirs for all tools
set(LOOM_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/affine/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/triton-shared/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/TileLoomToTTKernel/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/loom-opt/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/constraint-opt/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/cost-model/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)


# Triton-shared tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-opt/single_stage)

# TileLoom to TTKernel tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tileloom-to-ttkernel)

# Loom-constraint tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-constraint/single_stage)



# Dataflow-dialect tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataflow-dialect)

# Resource-system demos
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/resource-system)

# Custom LSP server
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-lsp-server)
