/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/grids.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "tensor_test.h"

using namespace std::string_literals;

std::filesystem::path make_temp_h5_path() {
  return std::filesystem::temp_directory_path() / std::filesystem::unique_path("grids_version_check_%%%%-%%%%-%%%%.h5");
}

struct file_cleanup_guard {
  explicit file_cleanup_guard(std::filesystem::path file_path) : path(std::move(file_path)) {}
  ~file_cleanup_guard() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  std::filesystem::path path;
};

inline std::pair<int, char**> get_argc_argv(std::string& str) {
  std::string        key;
  std::vector<char*> splits = {(char*)str.c_str()};
  for (int i = 1; i < str.size(); i++) {
    if (str[i] == ' ') {
      str[i] = '\0';
      splits.emplace_back(&str[++i]);
    }
  }
  char** argv = new char*[splits.size()];
  for (int i = 0; i < splits.size(); i++) {
    argv[i] = splits[i];
  }

  return {(int)splits.size(), argv};
}

void check_transformer(green::grids::transformer_t& tr) {
  size_t                   nk = 5;
  size_t                   ns = 2;
  size_t                   i  = 0;
  green::grids::dtensor<1> epsk(nk);
  std::transform(epsk.begin(), epsk.end(), epsk.begin(), [&i, nk](double x) { return std::cos((2.0 * M_PI * i++) / nk); });
  green::grids::ztensor<5> X1w;
  {
    green::grids::ztensor<3> tmp(tr.sd().repn_fermi().nw(), ns, nk);
    for (size_t iw = 0; iw < tmp.shape()[0]; ++iw) {
      tmp(iw, 0) += (1. / (tr.sd().repn_fermi().wsample()(iw) * 1.i - epsk));
      tmp(iw, 1) += (1. / (tr.sd().repn_fermi().wsample()(iw) * 1.i + epsk));
    }
    X1w = tmp.reshape(tr.sd().repn_fermi().nw(), ns, nk, 1, 1);
  }
  green::grids::ztensor<5> X1t(tr.sd().repn_fermi().nts(), ns, nk, 1, 1);
  green::grids::ztensor<5> X1w_b(tr.sd().repn_bose().nw(), ns, nk, 1, 1);
  green::grids::ztensor<5> X2w(tr.sd().repn_fermi().nw(), ns, nk, 1, 1);
  tr.omega_to_tau(X1w, X1t);
  tr.tau_to_omega(X1t, X2w);
  REQUIRE_THAT(X2w, IsCloseTo(X1w, 1e-10));

  // Test mixed grid transformation
  green::grids::ztensor<5> X2t(X1t.shape());
  tr.tau_f_to_w_b(X1t, X1w_b);
  tr.w_b_to_tau_f(X1w_b, X2t);
  REQUIRE_THAT(X2t, IsCloseTo(X1t, 1e-10));

  std::array<size_t, 5>    shape = X1w.shape();
  green::grids::ztensor<2> X3w(shape[3], shape[4]);
  green::grids::ztensor<3> X4w(shape[2], shape[3], shape[4]);

  for (int iw = 0; iw < shape[0]; ++iw) {
    for (int is = 0; is < shape[1]; ++is) {
      for (int ik = 0; ik < shape[2]; ++ik) {
        tr.tau_to_omega_wsk(X1t, X3w, iw, is, ik);
        REQUIRE_THAT(X3w, IsCloseTo(X1w(iw, is, ik)));
      }
      tr.tau_to_omega_ws(X1t, X4w, iw, is);
      REQUIRE_THAT(X4w, IsCloseTo(X1w(iw, is)));
    }
  }

  SECTION("FERMI-BOSON") {
    std::transform(epsk.begin(), epsk.end(), epsk.begin(), [&i, nk](double x) { return std::cos((2.0 * M_PI * i++) / nk); });
    green::grids::ztensor<4> X5w_f;
    {
      green::grids::ztensor<2> tmp(tr.sd().repn_fermi().nw(), nk);
      for (size_t iw = 0; iw < tmp.shape()[0]; ++iw) {
        tmp(iw) += (1. / (tr.sd().repn_fermi().wsample()(iw) * 1.i - epsk));
      }
      X5w_f = tmp.reshape(tr.sd().repn_fermi().nw(), nk, 1, 1);
    }
    green::grids::ztensor<4> X5t_f(tr.sd().repn_fermi().nts(), nk, 1, 1);
    tr.omega_to_tau(X5w_f, X5t_f);
    green::grids::ztensor<4> X5t_b(tr.sd().repn_bose().nts(), nk, 1, 1);
    green::grids::ztensor<4> X6t_f(X5t_f.shape());
    tr.fermi_boson_trans(X5t_f, X5t_b, 1);
    tr.fermi_boson_trans(X5t_b, X6t_f, 0);
    REQUIRE_THAT(X5t_f, IsCloseTo(X6t_f));
  }
  SECTION("Fixed Frequency Transform") {
    green::grids::ztensor<5> X1w_partial(2, ns, nk, 1, 1);
    tr.tau_f_to_w_b(X1t, X1w_b);
    tr.tau_f_to_w_b(X1t, X1w_partial, 2, 2);
    REQUIRE_THAT(X1w_b(2), IsCloseTo(X1w_partial(0)));
    REQUIRE_THAT(X1w_b(3), IsCloseTo(X1w_partial(1)));
  }
  SECTION("Fixed Time Transform") {
    green::grids::ztensor<5> X1t_2(tr.sd().repn_fermi().nts(), ns, nk, 1, 1);
    green::grids::ztensor<5> X1t_partial(2, ns, nk, 1, 1);
    green::grids::ztensor<5> X1t_full(tr.sd().repn_fermi().nts(), ns, nk, 1, 1);
    tr.w_b_to_tau_f(X1w_b, X1t_2);
    tr.w_b_to_tau_f(X1w_b, X1t_partial, 2, 2);
    tr.w_b_to_tau_f(X1w_b, X1t_full, 2, 2, true);
    REQUIRE_THAT(X1t_2(2), IsCloseTo(X1t_partial(0)));
    REQUIRE_THAT(X1t_2(3), IsCloseTo(X1t_partial(1)));
    REQUIRE_THAT(X1t_2(2), IsCloseTo(X1t_full(2)));
    REQUIRE_THAT(X1t_2(3), IsCloseTo(X1t_full(3)));
    REQUIRE_THAT(X1t_2(0), !IsCloseTo(X1t_full(0)));
  }
  SECTION("To And From Basis") {
    green::grids::ztensor<5> X1c(tr.sd().repn_fermi().ni(), ns, nk, 1, 1);
    green::grids::ztensor<5> X1w_2(X1w.shape());
    green::grids::ztensor<5> X1t_2(X1t.shape());
    green::grids::ztensor<5> X1t_3(1, ns, nk, 1, 1);
    tr.matsubara_to_chebyshev(X1w, X1c, 1);
    tr.chebyshev_to_matsubara(X1c, X1w_2, 1);
    REQUIRE_THAT(X1w, IsCloseTo(X1w_2));
    tr.tau_to_chebyshev(X1t, X1c, 1);
    tr.chebyshev_to_tau(X1c, X1t_2, 1);
    tr.chebyshev_to_tau(X1c, X1t_3, 1, true);

    REQUIRE_THAT(X1t, IsCloseTo(X1t_2));
    REQUIRE_THAT(X1t(tr.sd().repn_fermi().nts() - 1), IsCloseTo(X1t_3(0)));
  }
  SECTION("Bose") {
    green::grids::ztensor<4> W1w_b;
    {
      green::grids::ztensor<2> tmp(tr.sd().repn_bose().nw(), nk);
      for (size_t iw = 0; iw < tmp.shape()[0]; ++iw) {
        tmp(iw) += (1. / (tr.sd().repn_bose().wsample()(iw) * 1.i - epsk));
      }
      W1w_b = tmp.reshape(tr.sd().repn_bose().nw(), nk, 1, 1);
    }
    green::grids::ztensor<4> W1t_b(tr.sd().repn_bose().nts(), nk, 1, 1);
    green::grids::ztensor<4> W2t_b(tr.sd().repn_bose().nts(), nk, 1, 1);
    green::grids::ztensor<4> W3t_b(1, nk, 1, 1);
    green::grids::ztensor<4> W1c_b(tr.sd().repn_bose().ni(), nk, 1, 1);
    tr.omega_to_tau(W1w_b, W1t_b, 0);
    green::grids::ztensor<4> W2w_b(tr.sd().repn_bose().nw(), nk, 1, 1);
    green::grids::ztensor<4> W3w_b(1, nk, 1, 1);
    tr.tau_to_omega(W1t_b, W2w_b, 0);
    tr.tau_to_omega_w(W1t_b, W3w_b, 5, 0);
    REQUIRE_THAT(W1w_b, IsCloseTo(W2w_b));
    REQUIRE_THAT(W1w_b(5), IsCloseTo(W3w_b(0)));

    tr.tau_to_chebyshev(W1t_b, W1c_b, 0);
    tr.chebyshev_to_tau(W1c_b, W2t_b, 0);
    tr.chebyshev_to_tau(W1c_b, W3t_b, 0, true);

    REQUIRE_THAT(W1t_b, IsCloseTo(W2t_b));
    REQUIRE_THAT(W1t_b(tr.sd().repn_bose().nts() - 1), IsCloseTo(W3t_b(0)));
  }
  SECTION("Check Leakage") {
    double leakage = tr.check_chebyshev(X1t, 1);
    REQUIRE(leakage < 1e-10);
  }
  SECTION("Check Version Info") {
    std::string v = tr.get_version();
    std::string v2 = "0.2.0";  // Older version
    REQUIRE(green::grids::compare_version_strings(v, green::grids::GRIDS_MIN_VERSION) >= 0);
    REQUIRE(green::grids::compare_version_strings(v2, green::grids::GRIDS_MIN_VERSION) < 0);
  }
  SECTION("Check Version Consistency in HDF5 File") {
    // 1. Starting with new file (does not exist).
    //    Version check should pass because there's nothing to check / no inconsistency
    std::filesystem::path res_file_path = make_temp_h5_path();
    file_cleanup_guard cleanup(res_file_path);
    std::string res_file = res_file_path.string();
    REQUIRE_NOTHROW(green::grids::check_grids_version_in_hdf5(res_file, "0.2.4"));

    // 2. Open the file and set the __grids_version__ attribute to 0.2.4
    //    Then comparing with 0.2.4 should pass;
    //    comparing with older version should throw outdated_grids_file_error;
    //    comparing with newer version should throw outdated_results_file_error
    green::h5pp::archive ar_res_1(res_file, "w");
    ar_res_1.set_attribute<std::string>("__grids_version__", "0.2.4");
    ar_res_1.close();
    REQUIRE_NOTHROW(green::grids::check_grids_version_in_hdf5(res_file, green::grids::GRIDS_MIN_VERSION));
    REQUIRE_THROWS_AS(green::grids::check_grids_version_in_hdf5(res_file, "0.2.3"),
                      green::grids::outdated_grids_file_error);
    REQUIRE_THROWS_AS(green::grids::check_grids_version_in_hdf5(res_file, "0.2.5"),
                      green::grids::outdated_results_file_error);
  }
}

TEST_CASE("Grids") {
  SECTION("Chebyshev") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string grid_path  = GRID_PATH;
    std::string input_file = grid_path + "/cheb/100.h5"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);

    green::grids::transformer_t tr(p);
    REQUIRE(tr.sd().repn(0).repn() == "chebyshev");
    REQUIRE(tr.sd().repn(1).repn() == "chebyshev");
    check_transformer(tr);
  }

  SECTION("IR") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string grid_path  = GRID_PATH;
    std::string input_file = grid_path + "/ir/1e4.h5"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);

    green::grids::transformer_t tr(p);
    REQUIRE(tr.sd().repn(0).repn() == "ir");
    REQUIRE(tr.sd().repn(1).repn() == "ir");
    check_transformer(tr);
  }

  SECTION("File Does Not Exist") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string input_file = "XXX"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    REQUIRE_THROWS_AS(green::grids::transformer_t(p), green::grids::grids_file_not_found_error);
  }

  SECTION("Outdated grid - compatible") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string input_file = TEST_PATH + "/1e4_old_compatible.h5"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    REQUIRE_NOTHROW(green::grids::transformer_t(p));
  }

  SECTION("Outdated grid - incompatible") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string input_file = TEST_PATH + "/1e4_old_incompatible.h5"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    REQUIRE_THROWS_AS(green::grids::transformer_t(p), green::grids::outdated_grids_file_error);
  }

#ifndef NDEBUG
  SECTION("Wrong File") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string input_file = TEST_PATH + "/wrong_grid.h5"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    REQUIRE_THROWS_AS(green::grids::transformer_t(p), green::grids::grids_type_mismatch_error);
  }
#endif

  SECTION("Read Internally Stored Grids") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    std::string input_file = "ir/1e4.h5"s;
    std::string args       = "test --BETA 10 --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    green::grids::transformer_t tr(p);

    check_transformer(tr);
  }
}
