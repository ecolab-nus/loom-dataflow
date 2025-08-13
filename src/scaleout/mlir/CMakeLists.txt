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



# Analyzer that emits a Transform IR file to tile the first reduction by 8.
set(LINALG_TILE_REDUCE_TRANSFORM_SRC linalg_tile_reduce_transform.cpp)

add_executable(tmd_linalg_tile_reduce_transform ${LINALG_TILE_REDUCE_TRANSFORM_SRC})

target_include_directories(tmd_linalg_tile_reduce_transform PRIVATE ${MLIR_INCLUDE_DIRS})

target_link_libraries(tmd_linalg_tile_reduce_transform PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRPass
)

mlir_check_all_link_libraries(tmd_linalg_tile_reduce_transform)

