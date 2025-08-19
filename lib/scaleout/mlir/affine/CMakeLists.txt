set(AFFINE_ANALYZE_SRC
  affine_analyze_main.cpp
  syntax_check.cpp
  input_sharing_analysis.cpp
)
add_executable(tmd_affine_analyze ${AFFINE_ANALYZE_SRC})

target_include_directories(tmd_affine_analyze PRIVATE ${MLIR_INCLUDE_DIRS})

target_link_libraries(tmd_affine_analyze PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
)

mlir_check_all_link_libraries(tmd_affine_analyze)

