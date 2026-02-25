/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef GRIDS_COMMON_DEFS_H
#define GRIDS_COMMON_DEFS_H

#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/h5pp/archive.h>
#include "except.h"

#include <Eigen/Dense>
#include <cstdio>

using namespace std::string_literals;

namespace green::grids {
  // VERSION INFO
  inline const std::string GRIDS_MIN_VERSION = "0.2.4";

  // NDArray types
  template <size_t D>
  using itensor = green::ndarray::ndarray<int, D>;
  template <size_t D>
  using ltensor = green::ndarray::ndarray<long, D>;
  template <size_t D>
  using dtensor = green::ndarray::ndarray<double, D>;
  template <size_t D>
  using ztensor = green::ndarray::ndarray<std::complex<double>, D>;
  // Matrix types
  template <typename prec>
  using MatrixX   = Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcf = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // Matrix-Map types
  template <typename prec>
  using MMatrixX   = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcd = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcf = Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXd  = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // Const Matrix-Map types
  template <typename prec>
  using CMMatrixX   = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcf = Eigen::Map<const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXd  = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  inline static void define_parameters(params::params& p) {
    p.define<std::string>("TNL,grid_file", "Sparse imaginary time/frequency grid file name");
    p.define<double>("BETA", "Inverse temperature");
  }

  std::string grid_path(const std::string& path);

  inline int compare_version_strings(const std::string& v1, const std::string& v2) {
    int major_V1 = 0, minor_V1 = 0, patch_V1 = 0;
    int major_V2 = 0, minor_V2 = 0, patch_V2 = 0;

    char suffix_V1[32] = "";
    char suffix_V2[32] = "";

    int parsed_1 = std::sscanf(v1.c_str(), "%d.%d.%d%30s", &major_V1, &minor_V1, &patch_V1, suffix_V1);
    int parsed_2 = std::sscanf(v2.c_str(), "%d.%d.%d%30s", &major_V2, &minor_V2, &patch_V2, suffix_V2);

    if (parsed_1 < 3) {
      throw std::runtime_error("First version string (v1) failed to parse: '" + v1 + "'. Expected format: major.minor.patch[suffix]");
    }
    if (parsed_2 < 3) {
      throw std::runtime_error("Second version string (v2) failed to parse: '" + v2 + "'. Expected format: major.minor.patch[suffix]");
    }

    if (major_V1 != major_V2) {
      return major_V1 > major_V2 ? 1 : -1;
    }
    if (minor_V1 != minor_V2) {
      return minor_V1 > minor_V2 ? 1 : -1;
    }
    if (patch_V1 != patch_V2) {
      return patch_V1 > patch_V2 ? 1 : -1;
    }

    return 0;
  }

  /**
   * @brief Checks consistency between grid-file version used in current run vs. the version
   * used to generate the results file that is being (re-)started from.
   * This is to prevent users from accidentally restarting from a results file that was generated with
   * an older version of green-grids, which can lead to silent errors in the results.
   * 
   * 1. If the results file does not have a green-grids version attribute, treat it as having been
   *    generated with the baseline grids version (GRIDS_MIN_VERSION, currently 0.2.4) and check
   *    that the current grid-file version is not newer; otherwise an error is raised.
   * 2. If the results file has a green-grids version attribute, compare that version with the current
   *    grid-file version and distinguish older, equal, and newer cases, allowing only compatible
   *    combinations and throwing specific errors when there is a mismatch.
   * 
   * @param results_file - path to the results file that is being restarted from
   * @param grid_file_version - version of green-grids used in the current run (can be obtained from DysonSolver::get_grids_version())
   */
  void check_grids_version_in_hdf5(const std::string& results_file, const std::string& grid_file_version);

}  // namespace green::grids
#endif  // GRIDS_COMMON_DEFS_H
