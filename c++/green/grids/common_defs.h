/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef GRIDS_COMMON_DEFS_H
#define GRIDS_COMMON_DEFS_H

#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
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

  /**
   * @brief Check whether a version string satisfies the minimum required GRIDS version.
   *
   * Compares the given version string against the library's minimum supported
   * GRIDS version specified by `GRIDS_MIN_VERSION`.
   *
   * @param v Version string to check (e.g. "0.2.4").
   * @return true if v is greater than or equal to GRIDS_MIN_VERSION.
   * @return false if v is less than GRIDS_MIN_VERSION.
   * @throws outdated_grids_file_error if the format of v or GRIDS_MIN_VERSION is incorrect.
   */
  inline bool CheckVersion(const std::string& v) {
    int major_Vin = 0, minor_Vin = 0, patch_Vin = 0;
    int major_Vref = 0, minor_Vref = 0, patch_Vref = 0;
  
    char suffixV[32] = "";
    char suffixM[32] = "";
  
    int parsed_in = std::sscanf(v.c_str(), "%d.%d.%d%30s", &major_Vin, &minor_Vin, &patch_Vin, suffixV);
    int parsed_ref = std::sscanf(GRIDS_MIN_VERSION.c_str(), "%d.%d.%d%30s", &major_Vref, &minor_Vref, &patch_Vref, suffixM);

    if (parsed_in < 3 || parsed_ref < 3) {
      throw outdated_grids_file_error("Version string format is incorrect. Expected format: major.minor.patch[suffix]");
    }
  
    if (major_Vin != major_Vref) return major_Vin > major_Vref;
    if (minor_Vin != minor_Vref) return minor_Vin > minor_Vref;
    if (patch_Vin != patch_Vref) return patch_Vin > patch_Vref;
  
    // If numeric parts in version are all equal, do not worry about suffix
    // e.g., 0.2.4b10 has same integral format as 0.2.4
    return true;
  }

}  // namespace green::grids
#endif  // GRIDS_COMMON_DEFS_H
