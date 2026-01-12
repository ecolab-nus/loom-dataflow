# Loom Constraint System (LCS) library
# Provides constraint analysis and Presburger integration for Loom IR

add_library(loomConstraintSystem STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_set.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/analysis_engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_space_utils.cpp
)

target_include_directories(loomConstraintSystem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
    # Include the generated Loom dialect headers
    ${CMAKE_BINARY_DIR}/lib/loom-dialect/IR
)

# Link against MLIR libraries and Loom dialect
target_link_libraries(loomConstraintSystem PUBLIC
    MLIRIR
    MLIRSupport
    MLIRPresburger
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRBytecodeOpInterface
    LLVMSupport
    loomDialect
)

# Depend on the generated Loom dialect headers
add_dependencies(loomConstraintSystem
    LOOMLoomOpsIncGen
    LOOMLoomDialectIncGen
)

target_compile_options(loomConstraintSystem PRIVATE -fno-rtti)

