set(LINALG_LIST_OPS_SRC linalg_list_ops.cpp)

add_executable(tmd_linalg_list_ops ${LINALG_LIST_OPS_SRC})

target_include_directories(tmd_linalg_list_ops PRIVATE ${MLIR_INCLUDE_DIRS})

target_link_libraries(tmd_linalg_list_ops PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRPass
)

mlir_check_all_link_libraries(tmd_linalg_list_ops)


