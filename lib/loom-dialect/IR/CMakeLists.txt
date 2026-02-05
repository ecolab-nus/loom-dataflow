# Loom dialect build in IR/

set(LLVM_TARGET_DEFINITIONS LoomOps.td)
# Build a robust -I list for mlir-tblgen
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()
mlir_tablegen(LoomOps.h.inc -gen-op-decls ${TBLGEN_INCLUDES})
mlir_tablegen(LoomOps.cpp.inc -gen-op-defs ${TBLGEN_INCLUDES})
mlir_tablegen(LoomEnums.h.inc -gen-enum-decls ${TBLGEN_INCLUDES})
mlir_tablegen(LoomEnums.cpp.inc -gen-enum-defs ${TBLGEN_INCLUDES})
mlir_tablegen(LoomAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=loom ${TBLGEN_INCLUDES})
mlir_tablegen(LoomAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=loom ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMLoomOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LoomDialect.td)
mlir_tablegen(LoomDialect.h.inc -gen-dialect-decls -dialect=loom ${TBLGEN_INCLUDES})
mlir_tablegen(LoomDialect.cpp.inc -gen-dialect-defs -dialect=loom ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMLoomDialectIncGen)

# Types (Removed)

add_mlir_dialect_library(loomDialect
  LoomOps.cpp

  DEPENDS
  LOOMLoomOpsIncGen
  LOOMLoomDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRAffineDialect
)

target_include_directories(loomDialect PUBLIC ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

