cmake_minimum_required(VERSION 3.18)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(grids)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(GreenDeps)

add_green_dependency(green-h5pp)
add_green_dependency(green-ndarray)
add_green_dependency(green-params)

add_subdirectory(c++)
add_library(GREEN::GRIDS ALIAS grids)

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif ()
install(TARGETS grids DESTINATION bin)
INSTALL(DIRECTORY ${grids_SOURCE_DIR}/data/ DESTINATION shared)
INSTALL(DIRECTORY ${grids_SOURCE_DIR}/c++/green/ DESTINATION include/green)