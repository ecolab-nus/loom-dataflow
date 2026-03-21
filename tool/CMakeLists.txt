# Root tools for affine and loom-opt

# Common include dirs for all tools
set(LOOM_PASSES_INC
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/common/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/loom-opt/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/tt-opt/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/passes/LCS/inc
  ${CMAKE_CURRENT_LIST_DIR}/../lib/analysis/inc
  ${CMAKE_BINARY_DIR}/lib/dataflow-dialect/IR
  ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)


# Triton-shared tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-opt/single_stage)

# TT-opt tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tt-opt/single_stage)



# Dataflow-dialect tools
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataflow-dialect)

# Resource-system demos
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/resource-system)

# Custom LSP server
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/loom-lsp-server)
