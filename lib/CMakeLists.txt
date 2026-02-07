# ScaleOut library - aggregates all submodules
# This is the main library that combines resources, modules, passes, and dialect

# Build submodules
add_subdirectory(resources)
add_subdirectory(modules)
add_subdirectory(passes)
add_subdirectory(dataflow-dialect)
add_subdirectory(loom-dialect)
add_subdirectory(lcs)

# Aggregate library that combines resources and modules
# This is an interface library that aggregates loomResources and loomModules
add_library(scaleout INTERFACE)

# Link submodule libraries
target_link_libraries(scaleout INTERFACE
    loomResources
    loomModules
)