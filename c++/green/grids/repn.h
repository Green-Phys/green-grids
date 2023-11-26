/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef GRIDS_REPN_H
#define GRIDS_REPN_H

#include <green/params/params.h>

#include "common_defs.h"
#include "itime_mesh_t.h"

using namespace std::complex_literals;

namespace green::grids {

  class RepnBase {
  private:
    // Frequency samplings
    dtensor<1>   _wsample;
    // Tau sampling
    dtensor<1>   _tsample;
    // Fourier normalization factor
    double       _npw;
    // number of imaginary time points
    size_t       _nts;
    // number of sparse intermediate basis points (currently either IR or Chebyshev)
    size_t       _ni;
    // number of frequency points
    size_t       _nw;
    // tau meshes
    itime_mesh_t _tau_mesh;
    // statistics shift: 0 - bose, 1 - fermi
    int          _stat;
    // statistics
    std::string  _stat_name;
    // inverse temperature
    double       _beta;

  public:
    size_t              nts() const { return _nts; }
    size_t              ni() const { return _ni; }
    size_t              nw() const { return _nw; }
    const dtensor<1>&   wsample() const { return _wsample; }
    const dtensor<1>&   tsample() const { return _tsample; }
    const itime_mesh_t& tau_mesh() const { return _tau_mesh; }
    const std::string&  stat_name() const { return _stat_name; }

  public:
    RepnBase(const params::params& p, int stat) : _stat(stat), _stat_name(stat ? "fermi" : "bose"), _beta(p["BETA"]) {
      green::h5pp::archive ar(grid_path(p["grid_file"]));
      read_xsample(ar);
      read_wsample(ar);
      ar.close();
    }
    RepnBase(double b, int stat) : _stat(stat), _stat_name(stat ? "fermi" : "bose"), _beta(b) {}

    virtual ~RepnBase()                                = default;

    // type of the compact representation
    virtual std::string repn() const                   = 0;
    /// scale transformation from intermediate basis to tau with respect to inverse temperature
    virtual void        scale_fx(MatrixXd&) const      = 0;
    /// scale transformation from tau to intermediate basis with respect to inverse temperature
    virtual void        scale_fx_inv(MatrixXd&) const  = 0;
    /// scale transformation from intermediate basis to matsubara frequencies with respect to inverse temperature
    virtual void        scale_fw(MatrixXcd&) const     = 0;
    /// scale transformation from matsubara frequencies to intermediate basis with respect to inverse temperature
    virtual void        scale_fw_inv(MatrixXcd&) const = 0;

    /// inverse temperature
    double              beta() const { return _beta; }

  private:
    /**
     * Read frequency sampling points
     * @param ar - [INPUT] hdf5 archive with grid data
     */
    void read_wsample(const green::h5pp::archive& ar) {
      ltensor<1> nsample;
      ar[_stat_name + "/ngrid"] >> nsample;
      _wsample.resize(nsample.size());
      _wsample << scale_w(nsample);
      _nw = _wsample.size();
    }
    /**
     * Read imaginary time sampling points
     * @param ar - [INPUT] hdf5 archive with grid data
     */
    void read_xsample(const green::h5pp::archive& ar) {
      dtensor<1> xsample;
      ar[_stat_name + "/xgrid"] >> xsample;
      _ni = xsample.size();
      // add 0 and beta;
      _tsample.resize(xsample.size() + 2);
      std::copy(xsample.begin(), xsample.end(), _tsample.begin() + 1);
      _nts               = _tsample.size();
      _tsample(0)        = -1;
      _tsample(_nts - 1) = +1;
      _tsample << scale_x(_tsample);
    }
    dtensor<1> scale_w(const ltensor<1>& ngrid) const { return (2 * ngrid + _stat) * M_PI / beta(); }
    dtensor<1> scale_x(const dtensor<1>& xgrid) const { return (xgrid + 1) * beta() / 2.0; }
  };

  class IR : public RepnBase {
  public:
    explicit IR(const params::params& p, int stat) : RepnBase(p, stat) {}
    explicit IR(double beta, int stat) : RepnBase(beta, stat) {}
    std::string repn() const override { return "ir"; }
    void        scale_fx(MatrixXd& Ttc) const override { Ttc *= std::sqrt(2.0 / beta()); }
    void        scale_fx_inv(MatrixXd& Tct) const override { Tct *= std::sqrt(beta() / 2.0); }
    void        scale_fw(MatrixXcd& Tnc) const override { Tnc *= std::sqrt(beta()); }
    void        scale_fw_inv(MatrixXcd& Tcn) const override { Tcn *= std::sqrt(1.0 / beta()); }
  };

  class Chebysheb : public RepnBase {
  public:
    explicit Chebysheb(const params::params& p, int stat) : RepnBase(p, stat) {}
    explicit Chebysheb(double beta, int stat) : RepnBase(beta, stat) {}
    std::string repn() const override { return "chebyshev"; }
    void        scale_fx(MatrixXd& Ttc) const override {}
    void        scale_fx_inv(MatrixXd& Tct) const override {}
    void        scale_fw(MatrixXcd& Tnc) const override { Tnc *= beta() / 2.0; }
    void        scale_fw_inv(MatrixXcd& Tcn) const override { Tcn *= 2.0 / beta(); }
  };
}  // namespace green::grids
#endif  // GRIDS_REPN_H
