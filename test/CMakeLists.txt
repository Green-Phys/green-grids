project(grids_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release_
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(grids_test grids_test.cpp)
target_compile_definitions(grids_test PRIVATE GRID_PATH="${grids_SOURCE_DIR}/data" TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(grids_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::GRIDS)

include(CTest)
include(Catch)
catch_discover_tests(grids_test)