# Passes subdirectory
# Organizes all MLIR pass libraries

add_subdirectory(affine)
add_subdirectory(triton-shared)
add_subdirectory(TileLoomToTTKernel)
add_subdirectory(common)
add_subdirectory(loom-opt)
add_subdirectory(constraint-opt)

