cmake_minimum_required(VERSION 3.19...3.27)
project(actionet)
# if (CMAKE_VERSION VERSION_LESS 3.18)
#   set(DEV_MODULE Development)
# else()
#   set(DEV_MODULE Development.Module)
# endif()

set(DEV_MODULE Development.Module)
find_package(Python 3.11 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()





# Add nanobind subdirectory at the end
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/nanobind)

# Placeholder for your actual extension module (wrapper source)
nanobind_add_module(test src/wp.cpp)
# nanobind_add_module(my_ext NOMINSIZE my_ext.cpp) # NOMINSIZE optimizes for speed over size
