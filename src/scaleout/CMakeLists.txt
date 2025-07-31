# ScaleOut static library
add_library(scaleout STATIC
    scaleout.cpp
    scaleout.h
    resources/mem_port.cpp
    resources/mem_capacity.cpp
)

target_include_directories(scaleout PUBLIC . resources)