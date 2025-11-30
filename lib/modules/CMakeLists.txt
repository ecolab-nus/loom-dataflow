# Modules library
# Provides higher-level module models: Network, Chain, Mesh2D, Torus

add_library(tmdModules STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/network.cpp
)

target_include_directories(tmdModules PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

# Modules depend on resources
target_link_libraries(tmdModules PUBLIC
    tmdResources
    MLIRIR
)
mlir_check_all_link_libraries(tmdModules)

