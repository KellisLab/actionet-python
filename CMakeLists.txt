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

Optional optimizations
option(ACTIONET_ENABLE_OPTIMIZED "Enable architecture-specific optimizations" OFF)
if(ACTIONET_ENABLE_OPTIMIZED)
    if(NOT APPLE)
        add_compile_options(-march=native)
    endif()
endif()

# Find dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/libactionet")

# OpenMP detection (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    set(HAVE_OPENMP TRUE)
else()
    message(STATUS "OpenMP not found - building without parallel support")
    set(HAVE_OPENMP FALSE)
endif()

# libactionet setup
set(LIBACTIONET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/libactionet")
if(NOT EXISTS "${LIBACTIONET_DIR}/include")
    message(FATAL_ERROR
        "libactionet not found. Please initialize the submodule:\n"
        "  git submodule update --init --recursive"
    )
endif()

# Build libactionet as a static library
# Collect all source files
file(GLOB_RECURSE LIBACTIONET_SOURCES "${LIBACTIONET_DIR}/src/*.cpp")
add_library(libactionet_static STATIC ${LIBACTIONET_SOURCES})

# Armadillo headers (bundled with libactionet)
set(ARMADILLO_INCLUDE_DIR "${LIBACTIONET_DIR}/include/extern/armadillo")

# Find BLAS/LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Configure libactionet static library
target_include_directories(libactionet_static PRIVATE
    ${LIBACTIONET_DIR}/include
    ${LIBACTIONET_DIR}/include/extern
    ${LIBACTIONET_DIR}/include/extern/armadillo
    ${LIBACTIONET_DIR}/include/extern/gcem
)

# Add Accelerate framework headers for macOS (for cblas.h)
if(APPLE)
    target_include_directories(libactionet_static PRIVATE
        /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
    )
endif()

target_compile_definitions(libactionet_static PRIVATE
    ARMA_64BIT_WORD
    ARMA_DONT_USE_WRAPPER
    ARMA_USE_LAPACK
    ARMA_USE_BLAS
)

target_link_libraries(libactionet_static
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

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

target_link_libraries(_core PRIVATE
    libactionet_static
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

if(HAVE_OPENMP)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE ACTIONET_HAS_OPENMP)
endif()

# Platform-specific libraries
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(_core PRIVATE ${ACCELERATE_FRAMEWORK})
        # Add Accelerate framework include directory for cblas.h
        target_include_directories(_core PRIVATE
            /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
        )
    endif()
elseif(UNIX)
    # Try to find cholmod, but don't fail if not found
    find_library(CHOLMOD_LIB cholmod)
    if(CHOLMOD_LIB)
        target_link_libraries(_core PRIVATE ${CHOLMOD_LIB})
    else()
        message(WARNING "cholmod not found - some features may be limited")
    endif()
endif()

# Install
install(TARGETS _core LIBRARY DESTINATION actionet_py)
