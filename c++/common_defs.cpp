/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include "green/grids/common_defs.h"

#include "green/grids/except.h"

namespace green::grids {
  std::string grid_path(const std::string& path) {
    if (!std::filesystem::exists(path) and !std::filesystem::exists(GRID_SRC_PATH + "/"s + path)) {
      throw grids_file_not_found_error("Grid file " + path + " does not exists");
    }
    return std::filesystem::exists(path) ? path : GRID_SRC_PATH + "/"s + path;
  }
}  // namespace green::grids