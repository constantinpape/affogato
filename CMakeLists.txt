cmake_minimum_required(VERSION 3.10)

project(affogato)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# This should make the visbility warnings go away,
# but does not do it
# set(CMAKE_CXX_VISIBILITY_PREFIX default)


# set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)


##############################
# Build options
##############################

option(BUILD_PYTHON "Build python bindings with pybind11" ON)
option(BUILD_JS "Build javascript bindings with emscripten" OFF)

###############################
# Include system / conda libs
###############################

# find libraries - pthread
find_package(Threads)

# boost
find_package(Boost 1.63.0 CONFIG REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# xtensor
find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})


###############################
# Python-bindings
###############################

if(BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS NumPy Interpreter Development)
    include_directories(${Python_NumPy_INCLUDE_DIRS})
    
    find_package(xtensor-python REQUIRED)
    include_directories(${xtensor-python_INCLUDE_DIRS})
    
    find_package(pybind11 CONFIG REQUIRED)
    include_directories(${pybind11_INCLUDE_DIRS})
endif()


###############################
# Set-up and install
###############################

# find global headers
file(GLOB_RECURSE headers include/*.hxx)
file(GLOB_RECURSE headers ${CMAKE_INSTALL_PREFIX}/include/*.hxx)
include_directories(include)

# add subdirectories
add_subdirectory(src)

# install the headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/affogato"
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hxx"
  PATTERN "*.hpp"
  PATTERN "*.h"
)
