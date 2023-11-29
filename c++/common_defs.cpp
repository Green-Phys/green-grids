/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include "green/grids/common_defs.h"

#include "green/grids/except.h"

namespace green::grids {
  std::string grid_path(const std::string& path) {
    std::string src_path = GRID_SRC_PATH + "/"s + path;
    std::string install_path = GRID_INSTALL_PATH + "/"s + path;
    if (!std::filesystem::exists(path) and !std::filesystem::exists(src_path) and !std::filesystem::exists(install_path)) {
      throw grids_file_not_found_error("Grid file " + path + " does not exists");
    }
    return std::filesystem::exists(path) ? path : (std::filesystem::exists(src_path) ? src_path : install_path);
  }
}  // namespace green::grids