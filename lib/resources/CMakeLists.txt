# Resources library
# Provides primitive resource models: Memory, Ring, Chain

add_library(loomResources STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ring.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/chain.cpp
)

target_include_directories(loomResources PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Link MLIR IR for AffineMap support
target_link_libraries(loomResources PUBLIC MLIRIR)
mlir_check_all_link_libraries(loomResources)

