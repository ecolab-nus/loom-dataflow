# Constraint Optimization Passes Library
# Provides passes for polynomial constraint optimization

add_library(loomConstraintOptPasses STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_canonicalize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_factorize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_decompose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/constraint_linearize.cpp
)

target_include_directories(loomConstraintOptPasses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(loomConstraintOptPasses PUBLIC
    LoomPassesCommon
    MLIRIR
    MLIRParser
    MLIRSupport
    LLVMSupport
    loomDialect
)
mlir_check_all_link_libraries(loomConstraintOptPasses)

target_compile_options(loomConstraintOptPasses PRIVATE -fno-rtti)
