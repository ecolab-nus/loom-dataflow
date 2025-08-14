set(AFFINE_LIST_OPS_SRC affine_list_ops.cpp)

add_executable(tmd_affine_list_ops ${AFFINE_LIST_OPS_SRC})

target_include_directories(tmd_affine_list_ops PRIVATE ${MLIR_INCLUDE_DIRS})

target_link_libraries(tmd_affine_list_ops PRIVATE
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

mlir_check_all_link_libraries(tmd_affine_list_ops)


# Simple analysis pass tool that counts affine ops in a module
set(AFFINE_ANALYZE_SRC affine_analyze.cpp)
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


