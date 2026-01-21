# TileLoomToTTKernel pass library
# Provides conversion patterns from tiled Loom IR to TTKernel dialect

add_library(loomTileLoomToTTKernel STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/MemoryOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ComputeOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/AffineOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/FuncOpToTTKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/TileLoomToTTKernel.cpp
)

# Add tt-mlir include directories if available:
#   - Source headers:      /root/tt-mlir/include
#   - Generated .inc files: /root/tt-mlir/build/include
set(TTMLIR_INCLUDE_DIR "/root/tt-mlir/include")
set(TTMLIR_BUILD_INCLUDE_DIR "/root/tt-mlir/build/include")

if(NOT EXISTS "${TTMLIR_INCLUDE_DIR}")
  # Fallbacks for source includes
  find_path(TTMLIR_INCLUDE_DIR
    NAMES ttmlir/Dialect/TTKernel/IR/TTKernelOps.h
    PATHS
      ${CMAKE_SOURCE_DIR}/../tt-mlir/include
      /usr/local/include
      /opt/tt-mlir/include
  )
endif()

if(NOT EXISTS "${TTMLIR_BUILD_INCLUDE_DIR}")
  # Fallbacks for generated includes
  find_path(TTMLIR_BUILD_INCLUDE_DIR
    NAMES ttmlir/Dialect/TTKernel/IR/TTKernelOpsDialect.h.inc
    PATHS
      ${CMAKE_SOURCE_DIR}/../tt-mlir/build/include
      /usr/local/include
      /opt/tt-mlir/build/include
  )
endif()

target_include_directories(loomTileLoomToTTKernel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    ${TTMLIR_INCLUDE_DIR}
    ${TTMLIR_BUILD_INCLUDE_DIR}
)

# Add tt-mlir library directory and libraries
set(TTMLIR_LIB_DIR "/root/tt-mlir/build/lib")
if(EXISTS "${TTMLIR_LIB_DIR}")
    link_directories(${TTMLIR_LIB_DIR})
    # Link directly to the static libraries
    set(TTMLIR_TTKERNEL_LIB "${TTMLIR_LIB_DIR}/libMLIRTTKernelDialect.a")
    set(TTMLIR_TTCORE_LIB "${TTMLIR_LIB_DIR}/libMLIRTTCoreDialect.a")
    set(TTMLIR_TTMETAL_LIB "${TTMLIR_LIB_DIR}/libMLIRTTMetalDialect.a")
else()
    # Try to find libraries
    find_library(TTMLIR_TTKERNEL_LIB
        NAMES MLIRTTKernelDialect
        PATHS
            ${CMAKE_SOURCE_DIR}/../tt-mlir/build/lib
            /usr/local/lib
            /opt/tt-mlir/lib
    )
    find_library(TTMLIR_TTCORE_LIB
        NAMES MLIRTTCoreDialect
        PATHS
            ${CMAKE_SOURCE_DIR}/../tt-mlir/build/lib
            /usr/local/lib
            /opt/tt-mlir/lib
    )
    find_library(TTMLIR_TTMETAL_LIB
        NAMES MLIRTTMetalDialect
        PATHS
            ${CMAKE_SOURCE_DIR}/../tt-mlir/build/lib
            /usr/local/lib
            /opt/tt-mlir/lib
    )
endif()

# Link MLIR components
target_link_libraries(loomTileLoomToTTKernel PUBLIC
    MLIRIR
    MLIRPass
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTransforms
    MLIRSupport
    LLVMSupport
    # TTKernel dialect libraries from tt-mlir
    # TTKernel depends on TTCore, which depends on TTMetal
    ${TTMLIR_TTKERNEL_LIB}
    ${TTMLIR_TTCORE_LIB}
    ${TTMLIR_TTMETAL_LIB}
)
mlir_check_all_link_libraries(loomTileLoomToTTKernel)

target_compile_options(loomTileLoomToTTKernel PRIVATE -fno-rtti)
