/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef GRIDS_EXCEPT_H
#define GRIDS_EXCEPT_H
#include <stdexcept>

namespace green::grids {
  class grids_type_mismatch_error : public std::runtime_error {
  public:
    explicit grids_type_mismatch_error(const std::string& string) : runtime_error(string) {}
  };

  class grids_file_not_found_error : public std::runtime_error {
  public:
    explicit grids_file_not_found_error(const std::string& string) : runtime_error(string) {}
  };
}  // namespace green::grids

#endif  // GRIDS_EXCEPT_H
