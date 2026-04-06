# TileLoom to TTKernel mlir-opt wrapper

add_executable(tileloom_to_ttkernel_opt
  ${CMAKE_CURRENT_LIST_DIR}/tileloom_to_ttkernel_opt.cpp
)

# Add tt-mlir include directories
set(TTMLIR_INCLUDE_DIR "/root/tt-mlir/include")
set(TTMLIR_BUILD_INCLUDE_DIR "/root/tt-mlir/build/include")

target_include_directories(tileloom_to_ttkernel_opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_PASSES_INC}
  ${CMAKE_BINARY_DIR}/lib/adl-dialect/IR
  ${TTMLIR_INCLUDE_DIR}
  ${TTMLIR_BUILD_INCLUDE_DIR}
)

# Add tt-mlir library directory
set(TTMLIR_LIB_DIR "/root/tt-mlir/build/lib")
if(EXISTS "${TTMLIR_LIB_DIR}")
    link_directories(${TTMLIR_LIB_DIR})
    set(TTMLIR_TTKERNEL_LIB "${TTMLIR_LIB_DIR}/libMLIRTTKernelDialect.a")
    set(TTMLIR_TTCORE_LIB "${TTMLIR_LIB_DIR}/libMLIRTTCoreDialect.a")
    set(TTMLIR_TTMETAL_LIB "${TTMLIR_LIB_DIR}/libMLIRTTMetalDialect.a")
endif()

target_link_libraries(tileloom_to_ttkernel_opt PRIVATE
  loomTileLoomToTTKernel
  loomADLDialect
  loomDialect
  MLIRMlirOptMain
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  # TTKernel dialect libraries from tt-mlir
  ${TTMLIR_TTKERNEL_LIB}
  ${TTMLIR_TTCORE_LIB}
  ${TTMLIR_TTMETAL_LIB}
)

mlir_check_all_link_libraries(tileloom_to_ttkernel_opt)

target_compile_options(tileloom_to_ttkernel_opt PRIVATE -fno-rtti)

set_target_properties(tileloom_to_ttkernel_opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
