/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <fstream>
#include <string>

#include "green/grids/transformer_t.h"

namespace green::grids {
  void transformer_t::read_trans_statistics(green::h5pp::archive& tnl_file, int eta, MatrixXcd& Tnc_out, MatrixXcd& Tcn_out,
                                            MatrixXd& Ttc_out, MatrixXd& Tct_out) {
    const RepnBase& repn        = _sd.repn(eta);
    std::string     prefix      = repn.stat_name();
    size_t          nts         = repn.nts();
    // _Tnc = (nw, nl), _Ttc = (nx+2, nl), _Tct = (nl, nx), _Ttc_other = (nx_b+2, nl)
    size_t          ncheb       = repn.ni();
    size_t          nw          = repn.nw();

    Tnc_out                     = MatrixXcd::Zero(nw, ncheb);
    Ttc_out                     = MatrixXd::Zero(nts, ncheb);
    MatrixXd            Ttc_tmp = MatrixXd::Zero(ncheb, ncheb);
    MatrixXd            Tct_tmp = MatrixXd::Zero(ncheb, ncheb);
    std::vector<double> Tt1c_tmp(ncheb);
    std::vector<double> Tt1c_minus_tmp(ncheb);
    tnl_file[prefix + "/uwl"] >> Tnc_out.data();
    tnl_file[prefix + "/uxl"] >> Ttc_tmp.data();
    tnl_file[prefix + "/u1l_pos"] >> Tt1c_tmp.data();
    tnl_file[prefix + "/u1l_neg"] >> Tt1c_minus_tmp.data();
    Tcn_out = Tnc_out.completeOrthogonalDecomposition().pseudoInverse();
    Tct_out = Ttc_tmp.completeOrthogonalDecomposition().pseudoInverse();

    for (size_t ic = 0; ic < ncheb; ++ic) {
      Ttc_out(0, ic)       = Tt1c_minus_tmp[ic];
      Ttc_out(nts - 1, ic) = Tt1c_tmp[ic];
      for (size_t it = 0; it < ncheb; ++it) {
        Ttc_out(it + 1, ic) = Ttc_tmp(it, ic);
      }
    }
    repn.scale_fw(Tnc_out);
    repn.scale_fw_inv(Tcn_out);
    repn.scale_fx(Ttc_out);
    repn.scale_fx_inv(Tct_out);
  }

  void transformer_t::read_trans(const std::string& path) {
    green::h5pp::archive tnl_file(grid_path(path));

    // Read version info
    if (tnl_file.has_attribute("__grids_version__")) {
      std::string v_str = tnl_file.get_attribute<std::string>("__grids_version__");
      set_version(v_str);
      if (!CheckVersion(v_str)) {
        throw outdated_grids_file_error("The grids file version " + v_str +
                                        " is outdated. Minimum required version is " + GRIDS_MIN_VERSION + ".");
      }
    } else {
      set_version(GRIDS_MIN_VERSION);
    }

    read_trans_statistics(tnl_file, 1, _Tnc, _Tcn, _Ttc, _Tct);
    read_trans_statistics(tnl_file, 0, _Tnc_B, _Tcn_B, _Ttc_B, _Tct_B);

    MatrixXd Ttc_other_tmp   = MatrixXd::Zero(_sd.repn_bose().ni(), _sd.repn_fermi().ni());
    MatrixXd Ttc_B_other_tmp = MatrixXd::Zero(_sd.repn_fermi().ni(), _sd.repn_bose().ni());
    tnl_file["/fermi/uxl_other"] >> Ttc_other_tmp.data();
    tnl_file["/bose/uxl_other"] >> Ttc_B_other_tmp.data();

    // (_nts_b, _ncheb_b) * (_ncheb_b, _ncheb_b) * (_ncheb_b, _ncheb) = (_nts_b, nl)
    _Ttc_other   = _Ttc_B * _Tct_B * Ttc_other_tmp;
    _Ttc_B_other = _Ttc * _Tct * Ttc_B_other_tmp;
    tnl_file.close();

    _sd.repn_fermi().scale_fx(_Ttc_other);
    _sd.repn_bose().scale_fx(_Ttc_B_other);

    _Tnt    = _Tnc * _Tct;
    _Ttn    = _Ttc * _Tcn;
    _Tnt_B  = _Tnc_B * _Tct_B;
    _Ttn_B  = _Ttc_B * _Tcn_B;

    _Tnt_BF = _Tnc_B * _Tct_B * _Ttc_other.block(1, 0, _sd.repn_bose().ni(), _sd.repn_fermi().ni()) * _Tct;
    _Ttn_FB = _Ttc_B_other * _Tcn_B;
  }

  void transformer_t::fermi_boson_trans(const ztensor<4>& F_t_before, ztensor<4>& F_t_after, int eta) const {
    size_t dim_t = F_t_before.shape()[1] * F_t_before.shape()[2] * F_t_before.shape()[3];
    size_t dim_c = F_t_after.shape()[1] * F_t_after.shape()[2] * F_t_after.shape()[3];
    assert(dim_t == dim_c);

    if (eta == 1) {
      ztensor<4> F_c_before(_sd.repn_fermi().ni(), F_t_before.shape()[1], F_t_before.shape()[2], F_t_before.shape()[3]);
      tau_to_chebyshev(F_t_before, F_c_before, eta);
      // Transform to other stat's grid
      MMatrixXcd  f_t(F_t_after.data(), _sd.repn_bose().nts(), dim_t);
      // MMatrixXcd f_t(F_t_after.data()+dim1, _ncheb_b, dim1);
      CMMatrixXcd f_c(F_c_before.data(), F_c_before.shape()[0], dim_t);
      // _Ttc_other = (_nts_b, _ncheb)
      f_t = _Ttc_other * f_c;
    } else {
      ztensor<4> F_c_before(_sd.repn_bose().ni(), F_t_before.shape()[1], F_t_before.shape()[2], F_t_before.shape()[3]);
      tau_to_chebyshev(F_t_before, F_c_before, eta);
      // Transform to other stat's grid
      MMatrixXcd  f_t(F_t_after.data(), _sd.repn_fermi().nts(), dim_t);
      // MMatrixXcd f_t(F_t_after.data()+dim1, _ncheb, dim1);
      CMMatrixXcd f_c(F_c_before.data(), F_c_before.shape()[0], dim_t);
      // _Ttc_B_other = (_nts, ncheb_b)
      f_t = _Ttc_B_other * f_c;
    }
  }
}  // namespace green::grids
