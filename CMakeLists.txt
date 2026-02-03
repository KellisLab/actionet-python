cmake_minimum_required(VERSION 3.19)
project(actionet_python VERSION 0.1.0 LANGUAGES CXX)

# ==============================================================================
# Build Configuration
# ==============================================================================

# C++ standard (must match libactionet)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Platform-Specific Configuration
# ==============================================================================

if(APPLE)
    # macOS: Support macOS >= 11.0 for both x86_64 and arm64
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
    endif()

    # Architecture is typically set by scikit-build-core
    if(NOT CMAKE_OSX_ARCHITECTURES)
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE NATIVE_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(CMAKE_OSX_ARCHITECTURES ${NATIVE_ARCH} CACHE STRING "macOS architectures")
    endif()

    message(STATUS "Building for macOS ${CMAKE_OSX_DEPLOYMENT_TARGET}, arch: ${CMAKE_OSX_ARCHITECTURES}")

elseif(UNIX AND NOT APPLE)
    # Linux: manylinux2014 compatibility (glibc 2.17+)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

    # Ensure compatibility with older glibc
    add_compile_options(-fno-gnu-unique)

    message(STATUS "Building for Linux (manylinux2014 compatible)")
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

# Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Validate libactionet submodule
set(LIBACTIONET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/libactionet")
if(NOT EXISTS "${LIBACTIONET_DIR}/include/libactionet.hpp")
    message(FATAL_ERROR
        "libactionet submodule not initialized. Run:\n"
        "  git submodule update --init --recursive"
    )
endif()

# Build libactionet static library
# This configures BLAS, LAPACK, OpenMP, and Armadillo
add_subdirectory("${LIBACTIONET_DIR}")

# ==============================================================================
# Python Extension Module
# ==============================================================================

# Define Python wrapper sources
set(WRAPPER_SOURCES
    src/actionet/_core.cpp
    src/actionet/wp_utils.cpp
    src/actionet/wp_action.cpp
    src/actionet/wp_network.cpp
    src/actionet/wp_annotation.cpp
    src/actionet/wp_decomposition.cpp
    src/actionet/wp_tools.cpp
    src/actionet/wp_visualization.cpp
)

# Create pybind11 extension module
pybind11_add_module(_core MODULE ${WRAPPER_SOURCES})

# Set target properties
set_target_properties(_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Include directories
# Must include libactionet headers and its external dependencies (Armadillo, etc.)
target_include_directories(_core PRIVATE
    ${LIBACTIONET_DIR}/include
    ${LIBACTIONET_DIR}/include/extern
    ${LIBACTIONET_DIR}/include/extern/armadillo
)

# Link against libactionet
# This provides access to all libactionet functions and transitively links
# BLAS, LAPACK, and OpenMP (if available)
target_link_libraries(_core PRIVATE actionet)

# ==============================================================================
# Installation
# ==============================================================================

install(TARGETS _core
    LIBRARY DESTINATION actionet
    COMPONENT python_module
)
