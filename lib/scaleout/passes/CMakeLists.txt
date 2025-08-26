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

