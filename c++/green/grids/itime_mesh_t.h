/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GF2_ITIME_MESH_H
#define GF2_ITIME_MESH_H

#include <green/h5pp/archive.h>

#include <cmath>

#include "common_defs.h"

namespace green::grids {
  /**
   * @brief Imaginary-time mesh class defined at the Chebyshev nodes with additional boundary points
   */
  class itime_mesh_t {
    double              _beta;
    size_t              _ntau;
    std::vector<double> _points;

  public:
    itime_mesh_t() : _beta(0.0), _ntau(0) {}

    itime_mesh_t(const itime_mesh_t& rhs) : _beta(rhs._beta), _ntau(rhs._ntau), _points(rhs._points) {}

    itime_mesh_t(double beta, const std::vector<double>& points) : _beta(beta), _ntau(points.size()), _points(points) {}
    itime_mesh_t(double beta, const dtensor<1>& points) :
        _beta(beta), _ntau(points.size()), _points(points.begin(), points.end()) {}

    double                     operator[](size_t idx) const { return _points[idx]; }

    // Getter variables for members
    /// number of tau-points
    size_t                     extent() const { return _ntau; }

    /// inverse temperature
    double                     beta() const { return _beta; }

    /// vector of points
    const std::vector<double>& points() const { return _points; }

    /// Compare for equality
    bool                       operator==(const itime_mesh_t& mesh) const { return _beta == mesh._beta && _ntau == mesh._ntau; }

    /// Compare for non-equality
    bool                       operator!=(const itime_mesh_t& mesh) const { return !(*this == mesh); }
  };

  /// Stream output operator, e.g. for printing to file
  inline std::ostream& operator<<(std::ostream& os, const itime_mesh_t& M) {
    os << "# "
       << "IMAGINARY_TIME_CHEBYSHEV"
       << " mesh: N: " << M.extent() << " beta: " << M.beta() << std::endl;
    return os;
  }

}  // namespace green::grids

#endif  // GF2_ITIME_MESH_H
