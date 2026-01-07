# Common utilities for LOOM passes
# Provides reusable function cloning and enumeration utilities

add_library(LoomPassesCommon
  src/utils.cpp
)

target_include_directories(LoomPassesCommon
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(LoomPassesCommon
  PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRSupport
)

