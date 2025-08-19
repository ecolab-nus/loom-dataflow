# Dataflow (df) dialect build

set(LLVM_TARGET_DEFINITIONS DataflowOps.td)
# Build a robust -I list for mlir-tblgen
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()
mlir_tablegen(DataflowOps.h.inc -gen-op-decls ${TBLGEN_INCLUDES})
mlir_tablegen(DataflowOps.cpp.inc -gen-op-defs ${TBLGEN_INCLUDES})
add_public_tablegen_target(TMDDataflowOpsIncGen)

set(LLVM_TARGET_DEFINITIONS DataflowDialect.td)
mlir_tablegen(DataflowDialect.h.inc -gen-dialect-decls -dialect=df ${TBLGEN_INCLUDES})
mlir_tablegen(DataflowDialect.cpp.inc -gen-dialect-defs -dialect=df ${TBLGEN_INCLUDES})
add_public_tablegen_target(TMDDataflowDialectIncGen)

add_mlir_dialect_library(tmdDataflowDialect
  DataflowDialect.cpp
  DataflowOps.cpp

  DEPENDS
  TMDDataflowOpsIncGen
  TMDDataflowDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRAffineDialect
)

target_include_directories(tmdDataflowDialect PUBLIC ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})


