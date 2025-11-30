# ScaleOut library - aggregates all submodules
# This is the main library that combines resources, modules, passes, and dialect

# Build submodules
add_subdirectory(resources)
add_subdirectory(modules)
add_subdirectory(passes)
add_subdirectory(dataflow-dialect)

# Aggregate library that combines resources and modules
# This is an interface library that aggregates tmdResources and tmdModules
add_library(scaleout INTERFACE)

# Link submodule libraries
target_link_libraries(scaleout INTERFACE
    tmdResources
    tmdModules
)