# Modules library
# Provides higher-level module models: Network, Chain, Mesh2D, Torus

add_library(loomModules STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/network.cpp
)

target_include_directories(loomModules PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Modules depend on resources
target_link_libraries(loomModules PUBLIC
    loomResources
    MLIRIR
)
mlir_check_all_link_libraries(loomModules)

