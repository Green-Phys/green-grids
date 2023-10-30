/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GF2_FOURIER_TRANSFORM_H
#define GF2_FOURIER_TRANSFORM_H
#include <green/h5pp/archive.h>
#include <green/params/params.h>

#include "common_defs.h"
#include "itime_mesh_t.h"
#include "sparse_data.h"

namespace green::grids {

  /**
   * @brief Class for imaginary time transforms.
   * Perform Fourier transform between imaginary time and Matsubara frequency domains using intermediate representation.
   */
  class transformer_t {
  private:
    // Transformation matrices
    // Fermi
    // from ir basis to Matsubara
    MatrixXcd   _Tnc;
    // from Matsubara to ir
    MatrixXcd   _Tcn;
    // from ir to tau
    MatrixXd    _Ttc;
    // from fermionic ir to bosonic tau grid
    MatrixXd    _Ttc_other;
    // from tau to ir
    MatrixXd    _Tct;
    // Bose
    MatrixXcd   _Tnc_B;
    MatrixXcd   _Tcn_B;
    MatrixXd    _Ttc_B;
    // from bosonic ir to fermionic tau grid
    MatrixXd    _Ttc_B_other;
    MatrixXd    _Tct_B;

    MatrixXcd   _Tnt;
    MatrixXcd   _Ttn;
    MatrixXcd   _Tnt_B;
    MatrixXcd   _Ttn_B;

    // Transform from Bosonic frequency to Fermionic imaginary time
    MatrixXcd   _Ttn_FB;
    // Transform from Fermionic imaginary time to Bosonic frequency
    MatrixXcd   _Tnt_BF;

    sparse_data _sd;

  public:
    transformer_t(const green::params::params& p) : _sd(p) { read_trans(p["grid_file"]); }

    /**
     * @param n   - [INPUT] Matsubara frequency number, omega(n) = iw_n
     * @param eta - [INPUT] eta-parameter:
     *      0 - bosonic Matsubara
     *      1 - fermionic Matsubara
     *
     * @return value of w_n-th Matsubara frequency point
     */
    inline std::complex<double> omega(long n, int eta) const {
      return std::complex<double>(0.0, (2.0 * (n) + eta) * M_PI / _sd.beta());
    }
    /**
     * Read IR transformation matrix from IR representation to Matsubara axis
     * @param path   - [INPUT] path to Fermionic and Bosonic precomputed transformation matrices
     */
    void read_trans(const std::string& path);

    void read_trans_statistics(green::h5pp::archive& tnl_file, int eta, MatrixXcd& Tnc_out, MatrixXcd& Tcn_out, MatrixXd& Ttc_out,
                               MatrixXd& Tct_out);

    /**
     * Transformation between Fermionic and Bosonic grids.
     * @param eta - [INPUT] starting statistics
     */
    void fermi_boson_trans(const ztensor<4>& F_t_before, ztensor<4>& F_t_after, int eta);

    /**
     * Transform the tensor represented in fermionic imaginary time into bosonic Matsubara frequency through the intermediate
     * representation
     *
     * @tparam N  - tensor dimension
     * @param F_t - [INPUT]  tensor in fermionic imaginary time
     * @param F_w - [OUTPUT] tensor in bosonic Matsubara frequency
     */
    template <size_t N>
    void tau_f_to_w_b(const ztensor<N>& F_t, ztensor<N>& F_w) const {
      size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      size_t dim_w = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
      assert(dim_t == dim_w);

      MMatrixXcd  f_w(F_w.data(), F_w.shape()[0], dim_w);
      CMMatrixXcd f_t(F_t.data() + dim_t, _sd.repn_fermi().ni(), dim_t);
      f_w = _Tnt_BF * f_t;
    }

    template <size_t N>
    void tau_f_to_w_b(const ztensor<N>& F_t, ztensor<N>& F_w, size_t w, size_t n_w) const {
      size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      size_t dim_w = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
      assert(dim_t == dim_w);
      size_t      m   = _Tnt_BF.cols();
      MatrixXcd   Tnt = _Tnt_BF.block(w, 0, n_w, m);

      MMatrixXcd  f_w(F_w.data(), F_w.shape()[0], dim_w);
      CMMatrixXcd f_t(F_t.data() + dim_t, _sd.repn_fermi().ni(), dim_t);
      f_w = Tnt * f_t;
    }

    /**
     * Transform the tensor represented in bosonic Matsubara frequency into fermionic imaginary time through the intermediate
     * representation
     *
     * @tparam N  - tensor dimension
     * @param F_w - [INPUT] tensor in bosonic Matsubara frequency
     * @param F_t - [OUTPUT]  tensor in fermionic imaginary time
     */
    template <size_t N>
    void w_b_to_tau_f(const ztensor<N>& F_w, ztensor<N>& F_t) const {
      size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      size_t dim_w = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
      assert(dim_t == dim_w);
      MMatrixXcd  f_t(F_t.data(), F_t.shape()[0], dim_t);
      CMMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim_w);
      f_t = _Ttn_FB * f_w;
    }

    template <size_t N>
    void w_b_to_tau_f(const ztensor<N>& F_w, ztensor<N>& F_t, size_t w, size_t n_w) const {
      size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      size_t dim_w = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
      assert(dim_t == dim_w);

      size_t      m   = _Ttn_FB.rows();
      MatrixXcd   Ttn = _Ttn_FB.block(0, w, m, n_w);

      MMatrixXcd  f_t(F_t.data(), F_t.shape()[0], dim_t);
      CMMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim_w);
      f_t = Ttn * f_w;
    }

    /**
     * Intermediate step of non-uniform Fourier transformation. Convert object in Chebyshev/IR representation into Matsubara
     * frequency representation
     * @param F_c - [INPUT] Object in Chebyshev representation
     * @param F_w - [OUTPUT] Object in Matsubara frequency representation
     * @param eta - [INPUT] statistics
     */
    template <size_t N>
    void chebyshev_to_matsubara(const ztensor<N>& F_c, ztensor<N>& F_w, int eta) const {
      size_t      dim1 = std::accumulate(F_c.shape().begin() + 1, F_c.shape().end(), 1ul, std::multiplies<size_t>());
      // f_w(ncheb, dim1)
      MMatrixXcd  f_w(F_w.data(), F_w.shape()[0], dim1);
      // f_c(ncheb, dim1)
      CMMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim1);
      // Tnl(nwn=ncheb, ncheb)
      f_w = (eta ? _Tnc : _Tnc_B) * f_c;
    }

    /**
     * Intermediate step of non-uniform Fourier transformation. Convert object in Matsubara frequency representation into
     * Chebyshev/IR representation
     * @param F_w - [INPUT] Object in Matsubara frequency representation
     * @param F_c - [OUTPUT] Object in Chebyshev representation
     * @param eta - [INPUT] statistics
     */
    template <size_t N>
    void matsubara_to_chebyshev(const ztensor<N>& F_w, ztensor<N>& F_c, int eta) const {
      size_t      dim1 = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
      // f_c(ncheb, dim1)
      MMatrixXcd  f_c(F_c.data(), F_c.shape()[0], dim1);
      // f_w(nw, dim1)
      CMMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim1);
      // Tnl(nw, ncheb)
      f_c = (eta ? _Tcn : _Tcn_B) * f_w;
    }

    /**
     * Intermediate step of Chebyshev convolution. Convert object in Chebyshev/IR representation into tau axis
     */
    template <size_t N>
    void chebyshev_to_tau(const ztensor<N>& F_c, ztensor<N>& F_t, int eta, bool dm_only = false) const {
      size_t      dim_c = std::accumulate(F_c.shape().begin() + 1, F_c.shape().end(), 1ul, std::multiplies<size_t>());
      size_t      dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      // f_c(ncheb, dim_c)
      CMMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim_c);
      // f_t(nts, dim_t)
      MMatrixXcd  f_t(F_t.data(), F_t.shape()[0], dim_t);
      if (eta == 1) {
        if (!dm_only) {
          f_t = _Ttc * f_c;
        } else {
          f_t = _Ttc.row(_sd.repn_fermi().nts() - 1) * f_c;
        }
      } else {
        if (!dm_only) {
          f_t = _Ttc_B * f_c;
        } else {
          f_t = _Ttc_B.row(_sd.repn_bose().nts() - 1) * f_c;
        }
      }
    }

    /**
     * compute transform from imaginary time to intermediate basis for the input object.
     *
     * We use the following matrix multiplication scheme to go from imaginary time to Chebyshev polynomials
     * F_c(c, k, i, j) = _Fct(c, t) * F_t(t, k, i, j)
     *
     * Where _Fct is the transition matrix
     *
     * @param F_t - [INPUT] Object in imaginary time
     * @param F_c - [OUTPUT] Object in Chebyshev basis
     * @param eta - [INPUT] statistics
     */
    // template<size_t N>
    // void tau_to_chebyshev(const ztensor<N> &F_t, ztensor<N> &F_c, int eta = 1) const {
    //   tau_to_chebyshev(ztensor_view<N>(F_t), F_c, eta);
    // }
    template <size_t N>
    void tau_to_chebyshev(const ztensor<N>& F_t, ztensor<N>& F_c, int eta = 1) const {
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t     dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      // size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
      //  Chebyshev tensor as rectangular matrix
      MMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim1);
      // If using ir basis, _Tct = (nl, nx) instead of (nl, nx+2)
      if (eta == 1) {
        CMMatrixXcd f_t(F_t.data() + dim1, _sd.repn_fermi().ni(), dim1);
        f_c = _Tct * f_t;
      } else {
        CMMatrixXcd f_t(F_t.data() + dim1, _sd.repn_bose().ni(), dim1);
        f_c = _Tct_B * f_t;
      }
    }

    template <size_t N>
    void tau_to_chebyshev_c1_c2(const ztensor<N>& F_t, ztensor<N>& F_c, size_t ic1, size_t ic2, int eta = 1) const {
      size_t ni = eta == 1 ? _sd.repn_fermi().ni() : _sd.repn_bose().ni();
      assert(ic1 <= ic2);
      assert(ic2 < ni);
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t      dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      // size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
      //  Chebyshev tensor as rectangular matrix
      MMatrixXcd  f_c(F_c.data(), F_c.shape()[0], dim1);
      size_t      n   = (eta == 1 ? _Tct : _Tct_B).rows();
      size_t      m   = (eta == 1 ? _Tct : _Tct_B).cols();
      MatrixXcd   Tct = (eta == 1 ? _Tct : _Tct_B).block(ic1, 0, ic2 - ic1 + 1, n);
      // If using ir basis, _Tct = (nl, nx) instead of (nl, nx+2)
      CMMatrixXcd f_t(F_t.data() + dim1, ni, dim1);
      f_c = Tct * f_t;
    }

    template <size_t N>
    void tau_to_chebyshev_c(const ztensor<N>& F_t, ztensor<N>& F_c, size_t ic, int eta = 1) const {
      size_t ni = eta == 1 ? _sd.repn_fermi().ni() : _sd.repn_bose().ni();
      assert(ic < ni);
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t      dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      // size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
      //  Chebyshev tensor as rectangular matrix
      MMatrixXcd  f_c(F_c.data(), F_c.shape()[0], dim1);
      int         n   = (eta == 1 ? _Tct : _Tct_B).rows();
      int         m   = (eta == 1 ? _Tct : _Tct_B).cols();
      MatrixXcd   Tct = (eta == 1 ? _Tct : _Tct_B).block(ic, 0, 1, n);
      // If using ir basis, _Tct = (nl, nx) instead of (nl, nx+2)
      CMMatrixXcd f_t(F_t.data() + dim1, ni, dim1);
      f_c = Tct * f_t;
    }

    template <size_t N>
    void tau_to_omega(const ztensor<N>& F_t, ztensor<N>& F_w, int eta = 1) const {
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t     dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      // size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
      //  Chebyshev tensor as rectangular matrix
      MMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim1);
      if (eta == 1) {
        CMMatrixXcd f_t(F_t.data() + dim1, _sd.repn_fermi().ni(), dim1);
        f_w = _Tnt * f_t;
      } else {
        CMMatrixXcd f_t(F_t.data() + dim1, _sd.repn_bose().ni(), dim1);
        f_w = _Tnt_B * f_t;
      }
    }

    template <size_t N>
    void tau_to_omega_w(const ztensor<N>& F_t, ztensor<N>& F_w, size_t w, int eta = 1) const {
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());

      size_t n    = (eta == 1 ? _Tnt : _Tnt_B).rows();
      size_t m    = (eta == 1 ? _Tnt : _Tnt_B).cols();
      assert(w < n);
      MatrixXcd   Tnt = (eta == 1 ? _Tnt : _Tnt_B).block(w, 0, 1, m);

      MMatrixXcd  f_w(F_w.data(), F_w.shape()[0], dim1);
      CMMatrixXcd f_t(F_t.data() + dim1, m, dim1);
      f_w = Tnt * f_t;
    }

    template <size_t N>
    typename std::enable_if<(N > 3), void>::type tau_to_omega_wsk(const ztensor<N>& F_t, ztensor<N - 3>& F_w, size_t w, size_t s,
                                                                  size_t k, int eta = 1) const {
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      size_t dim2 = std::accumulate(F_t.shape().begin() + 2, F_t.shape().end(), 1ul, std::multiplies<size_t>());
      size_t dim3 = std::accumulate(F_t.shape().begin() + 3, F_t.shape().end(), 1ul, std::multiplies<size_t>());

      size_t n    = (eta == 1 ? _Tnt : _Tnt_B).rows();
      size_t m    = (eta == 1 ? _Tnt : _Tnt_B).cols();
      assert(w < n);
      MatrixXcd                                            Tnt = (eta == 1 ? _Tnt : _Tnt_B).block(w, 0, 1, m);

      MMatrixXcd                                           f_w(F_w.data(), 1, dim3);
      Eigen::Map<const MatrixXcd, 0, Eigen::OuterStride<>> f_t(F_t.data() + dim1 + s * dim2 + k * dim3, m, dim3,
                                                               Eigen::OuterStride<>(dim1));
      f_w = Tnt * f_t;
    }

    template <size_t N>
    void omega_to_tau(const ztensor<N>& F_w, ztensor<N>& F_t, int eta = 1) const {
      // Calculate coefficients in Chebyshev nodes
      // Dimension of the rest of arrays
      size_t     dim1 = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
      // size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
      //  Chebyshev tensor as rectangular matrix
      MMatrixXcd f_t(F_t.data(), F_t.shape()[0], dim1);
      if (eta == 1) {
        CMMatrixXcd f_w(F_w.data(), _sd.repn_fermi().nw(), dim1);
        f_t = _Ttn * f_w;
      } else {
        CMMatrixXcd f_w(F_w.data(), _sd.repn_bose().nw(), dim1);
        f_t = _Ttn_B * f_w;
      }
    }

    template <size_t D>
    inline void dump_intermediate_coeffs(ztensor<D>& F_c, std::string object, int iter) {
      size_t              ni   = F_c.shape()[0];
      size_t              rest = F_c.size() / ni;
      std::vector<double> max_coeffs(ni, 0.);

      for (size_t ic = 0; ic < ni; ++ic) {
        CMMatrixXcd F_c_m(F_c.data() + ic * rest, rest, 1);
        max_coeffs[ic] = F_c_m.cwiseAbs().maxCoeff();
      }

      green::h5pp::archive coeff_file("max_coeffs.h5", "a");
      coeff_file["iter" + std::to_string(iter) + "/" + object] << max_coeffs;
      coeff_file.close();
    }

    template <size_t N>
    double check_chebyshev(const ztensor<N>& F_t, int eta = 1) const {
      std::function<bool(const std::complex<double>&, const std::complex<double>&)> abs_comp(
          [](const std::complex<double>& lhs, const std::complex<double>& rhs) { return std::abs(lhs) < std::abs(rhs); });
      double                coeff_last  = -1.0;
      double                coeff_first = -1.0;
      std::array<size_t, N> shape       = F_t.shape();
      shape[0]                          = 1;
      ztensor<N> F_c(shape);
      tau_to_chebyshev_c(F_t, F_c, 0, eta);
      coeff_first = std::abs(*std::max_element(F_c.data(), F_c.data() + F_c.size(), abs_comp));
      shape[0]    = 2;
      F_c.resize(shape);
      size_t ni  = eta ? _sd.repn_fermi().ni() : _sd.repn_bose().ni();
      int    ic1 = ni - 2;
      int    ic2 = ni - 1;
      tau_to_chebyshev_c1_c2(F_t, F_c, ic1, ic2, eta);
      coeff_last = std::abs(*std::max_element(F_c.data(), F_c.data() + F_c.size(), abs_comp));
      return coeff_last / coeff_first;
    }

    /**
     * @return frequency samplings on Fermionic grid
     */
    dtensor<1>          wsample_fermi() const { return _sd.repn_fermi().wsample(); }

    /**
     * @return frequency samplings on Bosonic grid
     */
    dtensor<1>          wsample_bose() const { return _sd.repn_bose().wsample(); }

    const itime_mesh_t& tau_mesh() const { return _sd.repn_fermi().tau_mesh(); }

    /**
     * Getters
     */
  public:
    const MatrixXcd&   Ttn_FB() const { return _Ttn_FB; }

    const MatrixXcd&   Tnt_BF() const { return _Tnt_BF; }

    const MatrixXcd&   Tnc() const { return _Tnc; }

    const MatrixXcd&   Tcn() const { return _Tcn; }

    const MatrixXcd&   Tnc_B() const { return _Tnc_B; }

    const MatrixXd&    Ttc() const { return _Ttc; }

    const MatrixXd&    Ttc_B() const { return _Ttc_B; }

    const MatrixXd&    Tct() const { return _Tct; }

    const MatrixXcd&   Tnt() const { return _Tnt; }

    const MatrixXcd&   Ttn() const { return _Ttn; }

    const MatrixXcd&   Tnt_B() const { return _Tnt_B; }

    const MatrixXcd&   Ttn_B() const { return _Ttn_B; }

    const sparse_data& sd() const { return _sd; }
  };
}  // namespace green::grids

#endif  // GF2_FOURIER_TRANSFORM_H
