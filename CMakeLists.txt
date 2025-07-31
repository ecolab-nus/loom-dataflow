cmake_minimum_required(VERSION 3.14)

# Project name and version
project(tmd VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add subdirectories for static libraries
add_subdirectory(src/scaleout)
add_subdirectory(src/scalein)

# Create main executable from bin/main.cpp
if(EXISTS "${CMAKE_SOURCE_DIR}/bin/main.cpp")
    add_executable(${PROJECT_NAME} bin/main.cpp)
    target_link_libraries(${PROJECT_NAME} 
        scaleout 
        scalein
    )
    target_include_directories(${PROJECT_NAME} PRIVATE 
        src
    )
endif()

# Create resource management demo executable
if(EXISTS "${CMAKE_SOURCE_DIR}/bin/resource_demo.cpp")
    add_executable(${PROJECT_NAME}_resource_demo bin/resource_demo.cpp)
    target_link_libraries(${PROJECT_NAME}_resource_demo 
        scaleout
    )
    target_include_directories(${PROJECT_NAME}_resource_demo PRIVATE 
        src
    )
endif()

# GoogleTest setup
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Find all test files
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

if(TEST_SOURCES)
    # Create test executable
    add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
    
    # Link against GoogleTest and the static libraries
    target_link_libraries(${PROJECT_NAME}_tests 
        gtest_main
        gmock_main
        scaleout
        scalein
    )
    
    target_include_directories(${PROJECT_NAME}_tests PRIVATE 
        src
    )
    
    # Add tests to CTest
    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME}_tests)
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")