/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/grids.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

#include "tensor_test.h"

using namespace std::string_literals;

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

  for (int iw = 0; iw < shape[0]; ++iw) {
    for (int is = 0; is < shape[1]; ++is) {
      for (int ik = 0; ik < shape[2]; ++ik) {
        tr.tau_to_omega_wsk(X1t, X3w, iw, is, ik);
        REQUIRE_THAT(X3w, IsCloseTo(X1w(iw, is, ik)));
      }
    }
  }
}

TEST_CASE("Grids") {
  SECTION("Chebyshev") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    p.define<double>("BETA", "Inverse temperature", 10.0);
    std::string grid_path  = GRID_PATH;
    std::string input_file = grid_path + "/cheb/100.h5"s;
    std::string args       = "test --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);

    green::grids::transformer_t tr(p);

    check_transformer(tr);
  }

  SECTION("IR") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    p.define<double>("BETA", "Inverse temperature", 10.0);
    std::string grid_path  = GRID_PATH;
    std::string input_file = grid_path + "/ir/1e4.h5"s;
    std::string args       = "test --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);

    green::grids::transformer_t tr(p);

    check_transformer(tr);
  }
}
