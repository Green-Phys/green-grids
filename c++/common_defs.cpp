/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/h5pp/archive.h>
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

  void check_grids_version_in_hdf5(const std::string& results_file, const std::string& grid_file_version) {
    // If results file does not exist, nothing to check
    if (!std::filesystem::exists(results_file)) return;

    h5pp::archive ar(results_file, "r");
    if (ar.has_attribute("__grids_version__")) {
      std::string grids_version_in_results;
      grids_version_in_results = ar.get_attribute<std::string>("__grids_version__");
      ar.close(); // safely close before throwing
      if (compare_version_strings(grid_file_version, grids_version_in_results) < 0) {
        throw outdated_grids_file_error("The current green-grids version (" + grid_file_version +
                                          ") is older than the green-grids version used to create the original results file ("
                                          + grids_version_in_results +
                                          "). Please update green-grids to version " + grids_version_in_results);
      } else if (compare_version_strings(grid_file_version, grids_version_in_results) > 0) {
        throw outdated_results_file_error("The green-grids version used to create the results file (" + grids_version_in_results +
                                          ") is older than the current grid-grids version (" + grid_file_version +
                                          "). Please download the appropriate version from: " +
                                          "https://github.com/Green-Phys/green-grids/releases/ or https://github.com/Green-Phys/green-grids/tags");
      }
    } else {
      ar.close(); // safely close comparing version strings
      if (compare_version_strings(grid_file_version, GRIDS_MIN_VERSION) > 0) {
        throw outdated_results_file_error("The results file was created using un-versioned grid file (equiv. to " + GRIDS_MIN_VERSION +
                                          ") and the current green-grids version (" + grid_file_version + ") is newer.\n" + 
                                          "Please use old grid files from: https://github.com/Green-Phys/green-grids/releases/tag/v0.2.4.");
      }
    }
  }
}  // namespace green::grids