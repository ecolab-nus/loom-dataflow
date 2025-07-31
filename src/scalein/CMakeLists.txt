# ScaleIn static library
add_library(scalein STATIC
    scalein.cpp
    scalein.h
)

target_include_directories(scalein PUBLIC .)