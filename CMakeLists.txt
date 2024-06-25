cmake_minimum_required(VERSION 3.18)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(grids)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

################ GREEN RELEASE VERSION #####################
set(GREEN_RELEASE "v0.2.4")

set(CMAKE_CXX_STANDARD 17)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE TMP_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if ( PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME )
    set(GIT_HASH "${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")
else ()
    set(GIT_HASH "${GIT_HASH}**${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")
endif ()

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
install(TARGETS grids DESTINATION lib)
INSTALL(DIRECTORY ${grids_SOURCE_DIR}/data/ DESTINATION share)
INSTALL(DIRECTORY ${grids_SOURCE_DIR}/c++/green/ DESTINATION include/green)