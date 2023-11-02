/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef GRIDS_SPARSE_DATA_H
#define GRIDS_SPARSE_DATA_H

#include <green/h5pp/archive.h>
#include <green/params/params.h>

#include "common_defs.h"
#include "except.h"
#include "repn.h"

namespace green::grids {
  class sparse_data {
  public:
    sparse_data(const params::params& p) : _beta(p["BETA"]) { read_representation(p); }

    const RepnBase& repn(int stat) const { return stat ? *_repn_fermi.get() : *_repn_bose.get(); }

    const RepnBase& repn_fermi() const { return *_repn_fermi.get(); }
    const RepnBase& repn_bose() const { return *_repn_bose.get(); }

    double          beta() const { return _beta; }

  private:
    void read_representation(const params::params& p) {
      std::string   type_fermi;
      std::string   type_bose;
      h5pp::archive ar(p["grid_file"], "r");
      ar["fermi/metadata/type"] >> type_fermi;
      ar["bose/metadata/type"] >> type_bose;
      ar.close();
#ifndef NDEBUG
      if (type_fermi != type_bose) {
        throw grids_type_mismatch_error("Fermi and Bose grid representations should be of the same type");
      }
#endif
      _repn_fermi = init_repn(type_fermi, p, 1);
      _repn_bose  = init_repn(type_bose, p, 0);
    }

    std::unique_ptr<RepnBase> init_repn(const std::string& repn_type, const params::params& p, int stat) {
      if (repn_type == "ir") {
        return std::make_unique<IR>(p, stat);
      } else {
        return std::make_unique<Chebysheb>(p, stat);
      }
    }

  private:
    // Representations
    std::unique_ptr<RepnBase> _repn_fermi;
    std::unique_ptr<RepnBase> _repn_bose;

    // inverse temperature
    double                    _beta;
  };
}  // namespace green::grids
#endif  // GRIDS_SPARSE_DATA_H
