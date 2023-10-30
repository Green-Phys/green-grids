project(grids-lib CXX)

find_package(Eigen3 REQUIRED)

add_library(grids transformer.cpp)
target_include_directories(grids PUBLIC .)
target_link_libraries(grids PUBLIC GREEN::PARAMS GREEN::H5PP GREEN::NDARRAY)