# ScaleOut static library
add_library(scaleout STATIC
    scaleout.cpp
    scaleout.h
    resources/memory.cpp
    resources/ring.cpp
    resources/chain.cpp
    modules/network.cpp
)

target_include_directories(scaleout PUBLIC . resources modules ${MLIR_INCLUDE_DIRS})

# Link MLIR IR to provide headers and symbols for AffineMap used in modules
target_link_libraries(scaleout PUBLIC MLIRIR)
mlir_check_all_link_libraries(scaleout)

# Add standalone MLIR dialect
message(STATUS "Adding standalone MLIR dialect to scaleout")
add_subdirectory(standalone)
add_subdirectory(mlir)