add_library(loomDialectTransforms STATIC
  bufferizable_op_interface_impl.cpp
)

target_include_directories(loomDialectTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/../IR
)

target_link_libraries(loomDialectTransforms PUBLIC
  loomDialect
  MLIRIR
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
)

target_compile_options(loomDialectTransforms PRIVATE -fno-rtti)
