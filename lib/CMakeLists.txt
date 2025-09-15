# ScaleOut static library
add_library(scaleout STATIC
    resources/memory.cpp
    resources/ring.cpp
    resources/chain.cpp
    modules/network.cpp
)

# Link MLIR IR to provide headers and symbols for AffineMap used in modules
target_link_libraries(scaleout PUBLIC MLIRIR)
mlir_check_all_link_libraries(scaleout)

# Add MLIR components
add_subdirectory(passes)

# Dataflow dialect (df)
add_subdirectory(dataflow-dialect)

# Ensure the main library depends on the dialect so it builds in the project
target_link_libraries(scaleout PUBLIC tmdDataflowDialect)