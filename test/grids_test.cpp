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

TEST_CASE("Grids") {
  SECTION("Initialization") {}

  SECTION("Chebyshev") {}

  SECTION("IR") {
    auto p = green::params::params("DESCR");
    green::grids::define_parameters(p);
    p.define<double>("BETA", "Inverse temperature", 100.0);
    std::string input_file = GRID_PATH + "/ir/1e4.h5"s;
    std::string args       = "test --grid_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);

    green::grids::transformer_t tr(p);

    std::string                 test_path = TEST_PATH;

    green::grids::ztensor<5>    X1t;
    green::h5pp::archive        ar(test_path + "/test.h5", "r");
    ar["X_tau/data"] >> X1t;
    ar.close();
    green::grids::ztensor<5> X2t(X1t.shape());
    std::array<size_t, 5>    shape = X1t.shape();
    shape[0]                       = tr.sd().repn_fermi().ni();
    green::grids::ztensor<5> X1w(shape);
    green::grids::ztensor<2> X2w(shape[3], shape[4]);

    tr.tau_to_omega(X1t, X1w);
    for (int iw = 0; iw < shape[0]; ++iw) {
      for (int is = 0; is < shape[1]; ++is) {
        for (int ik = 0; ik < shape[2]; ++ik) {
          tr.tau_to_omega_wsk(X1t, X2w, iw, is, ik);
          REQUIRE_THAT(X2w, IsCloseTo(X1w(iw, is, ik)));
        }
      }
    }
    tr.omega_to_tau(X1w, X2t);
    REQUIRE_THAT(X2t, IsCloseTo(X1t, 1e-7));
  }
}
