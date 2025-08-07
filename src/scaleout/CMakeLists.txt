# ScaleOut static library
add_library(scaleout STATIC
    scaleout.cpp
    scaleout.h
    resources/mem_port.cpp
    resources/mem_capacity.cpp
)

target_include_directories(scaleout PUBLIC . resources)

# Add standalone MLIR dialect
message(STATUS "Adding standalone MLIR dialect to scaleout")
add_subdirectory(standalone)