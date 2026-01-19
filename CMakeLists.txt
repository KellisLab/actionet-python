cmake_minimum_required(VERSION 3.19)
project(actionet_py)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Platform-specific settings
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    
    # Set deployment target (CMake will handle -mmacosx-version-min automatically)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
    endif()

    # Set architecture (default to native, can be overridden)
    if(NOT CMAKE_OSX_ARCHITECTURES)
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(CMAKE_OSX_ARCHITECTURES ${ARCH})
    endif()
elseif(UNIX)
    # Linux: manylinux2014 compatibility
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Optional optimizations
option(ACTIONET_ENABLE_OPTIMIZED "Enable architecture-specific optimizations" OFF)
if(ACTIONET_ENABLE_OPTIMIZED)
    if(NOT APPLE)
        add_compile_options(-march=native)
    endif()
endif()

# Find dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# libactionet setup
set(LIBACTIONET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/libactionet")
if(NOT EXISTS "${LIBACTIONET_DIR}/include")
    message(FATAL_ERROR
        "libactionet not found. Please initialize the submodule:\n"
        "  git submodule update --init --recursive"
    )
endif()

# Build libactionet using its own build system
add_subdirectory("${LIBACTIONET_DIR}")

# OpenMP detection (optional) - use libactionet's detection
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found for Python wrapper: ${OpenMP_CXX_VERSION}")
    set(HAVE_OPENMP TRUE)
else()
    message(STATUS "OpenMP not found for Python wrapper - building without parallel support")
    set(HAVE_OPENMP FALSE)
endif()

# Armadillo headers (bundled with libactionet)
set(ARMADILLO_INCLUDE_DIR "${LIBACTIONET_DIR}/include/extern/armadillo")



# Create extension module
pybind11_add_module(_core MODULE
    src/actionet/_core.cpp
    src/actionet/wp_utils.cpp
    src/actionet/wp_action.cpp
    src/actionet/wp_network.cpp
    src/actionet/wp_annotation.cpp
    src/actionet/wp_decomposition.cpp
    src/actionet/wp_tools.cpp
    src/actionet/wp_visualization.cpp
)

target_include_directories(_core PRIVATE
    ${LIBACTIONET_DIR}/include
    ${ARMADILLO_INCLUDE_DIR}
)

target_compile_definitions(_core PRIVATE
    ARMA_64BIT_WORD
    ARMA_DONT_USE_WRAPPER
    ARMA_USE_LAPACK
    ARMA_USE_BLAS
)

# Link against the actionet library from the submodule
target_link_libraries(_core PRIVATE actionet)

if(HAVE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE ACTIONET_HAS_OPENMP)
endif()

# Platform-specific libraries (may already be linked via actionet target)
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        # Accelerate is likely already linked via actionet, but ensure headers are available
        target_include_directories(_core PRIVATE
            /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers
        )
    endif()
endif()

# Install
install(TARGETS _core LIBRARY DESTINATION actionet)

