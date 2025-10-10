# Resource-system demos

add_executable(resource_module_demo
  ${CMAKE_CURRENT_LIST_DIR}/module_demo.cpp
)

target_link_libraries(resource_module_demo PRIVATE
  scaleout
)

target_include_directories(resource_module_demo PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../lib
)

set_target_properties(resource_module_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(resource_demo
  ${CMAKE_CURRENT_LIST_DIR}/resource_demo.cpp
)

target_link_libraries(resource_demo PRIVATE
  scaleout
)

target_include_directories(resource_demo PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../lib
)

set_target_properties(resource_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


