# ADL dialect build in IR/

set(LLVM_TARGET_DEFINITIONS ADLOps.td)
# Build a robust -I list for mlir-tblgen
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()
mlir_tablegen(ADLOps.h.inc -gen-op-decls ${TBLGEN_INCLUDES})
mlir_tablegen(ADLOps.cpp.inc -gen-op-defs ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMADLOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ADLDialect.td)
mlir_tablegen(ADLDialect.h.inc -gen-dialect-decls -dialect=adl ${TBLGEN_INCLUDES})
mlir_tablegen(ADLDialect.cpp.inc -gen-dialect-defs -dialect=adl ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMADLDialectIncGen)

# Types
set(LLVM_TARGET_DEFINITIONS ADLTypes.td)
mlir_tablegen(ADLTypes.h.inc -gen-typedef-decls -typedefs-dialect=adl ${TBLGEN_INCLUDES})
mlir_tablegen(ADLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=adl ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMADLTypesIncGen)

add_mlir_dialect_library(loomADLDialect
  ADLDialect.cpp

  DEPENDS
  LOOMADLOpsIncGen
  LOOMADLDialectIncGen
  LOOMADLTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(loomADLDialect PUBLIC ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
