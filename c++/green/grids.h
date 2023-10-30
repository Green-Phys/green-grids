/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef GRIDS_GRIDS_H
#define GRIDS_GRIDS_H
#include <green/grids/common_defs.h>
#include <green/grids/except.h>
#include <green/grids/itime_mesh_t.h>
#include <green/grids/sparse_data.h>
#include <green/grids/transformer_t.h>

namespace green::grids {
  inline static void define_parameters(params::params& p) {
    p.define<std::string>("TNL,grid_file", "Sparse imaginary time/frequency grid file name");
  }

}  // namespace green::grids

#endif  // GRIDS_GRIDS_H
