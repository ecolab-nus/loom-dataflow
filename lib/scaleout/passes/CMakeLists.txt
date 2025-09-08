set(AFFINE_ANALYZE_SRC
  affine_analyze_main.cpp
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
  tmdDataflowDialect
)

mlir_check_all_link_libraries(tmd_affine_analyze)


# Exploration executable to enumerate spatial tilings of affine.parallel
add_executable(tmd_affine_explore
  affine_explore_main.cpp
  spatial_mapping.cpp
  affine_tile.cpp
)
target_include_directories(tmd_affine_explore PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(tmd_affine_explore PRIVATE
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
  tmdDataflowDialect
)
mlir_check_all_link_libraries(tmd_affine_explore)

add_dependencies(tmd_affine_explore
  TMDDataflowDialectIncGen
  TMDDataflowOpsIncGen
  TMDDataflowTypesIncGen
)

target_compile_options(tmd_affine_explore PRIVATE -fno-rtti)

# Tiling executable to transform affine.parallel with a given factor
add_executable(tmd_affine_tile
  affine_tile_main.cpp
  affine_tile.cpp
  affine_tile_pass.cpp
)
target_include_directories(tmd_affine_tile PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(tmd_affine_tile PRIVATE
  MLIRIR
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  LLVMSupport
)
target_compile_options(tmd_affine_tile PRIVATE -fno-rtti)
mlir_check_all_link_libraries(tmd_affine_tile)


## Triton-shared exploration executable
add_executable(tmd_triton_shared_explore
  triton_shared_explore_main.cpp
  spatial_mapping.cpp
  affine_tile.cpp
)

target_include_directories(tmd_triton_shared_explore PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(tmd_triton_shared_explore PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  tmdDataflowDialect
)
mlir_check_all_link_libraries(tmd_triton_shared_explore)

add_dependencies(tmd_triton_shared_explore
  TMDDataflowDialectIncGen
  TMDDataflowOpsIncGen
  TMDDataflowTypesIncGen
)

target_compile_options(tmd_triton_shared_explore PRIVATE -fno-rtti)


# Triton-shared spatial reuse analysis executable
add_executable(tmd_triton_shared_reuse
  triton_shared_reuse_main.cpp
  triton_shared_reuse_analysis.cpp
)

target_include_directories(tmd_triton_shared_reuse PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(tmd_triton_shared_reuse PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  tmdDataflowDialect
)
mlir_check_all_link_libraries(tmd_triton_shared_reuse)

add_dependencies(tmd_triton_shared_reuse
  TMDDataflowDialectIncGen
  TMDDataflowOpsIncGen
  TMDDataflowTypesIncGen
)

target_compile_options(tmd_triton_shared_reuse PRIVATE -fno-rtti)


# Triton-shared affinization executable
add_executable(tmd_triton_shared_affinize
  triton_shared_affinize_main.cpp
  triton_shared_affinize.cpp
)

target_include_directories(tmd_triton_shared_affinize PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(tmd_triton_shared_affinize PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRPass
  MLIRAsmParser
  MLIRBytecodeReader
  tmdDataflowDialect
)
mlir_check_all_link_libraries(tmd_triton_shared_affinize)

add_dependencies(tmd_triton_shared_affinize
  TMDDataflowDialectIncGen
  TMDDataflowOpsIncGen
  TMDDataflowTypesIncGen
)

target_compile_options(tmd_triton_shared_affinize PRIVATE -fno-rtti)
