add_executable(loom-lsp-server
  loom-lsp-server.cpp
)

target_include_directories(loom-lsp-server PRIVATE 
  ${MLIR_INCLUDE_DIRS} 
  ${LOOM_PASSES_INC}
)

target_link_libraries(loom-lsp-server PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRRegisterAllDialects
  MLIRLspServerLib
  loomDataflowDialect
  loomDialect
)

mlir_check_all_link_libraries(loom-lsp-server)

target_compile_options(loom-lsp-server PRIVATE -fno-rtti)
