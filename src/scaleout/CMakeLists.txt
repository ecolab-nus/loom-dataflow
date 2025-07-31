# ScaleOut static library
add_library(scaleout STATIC
    scaleout.cpp
    scaleout.h
)

target_include_directories(scaleout PUBLIC .)