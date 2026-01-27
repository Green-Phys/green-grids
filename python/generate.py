import argparse

import green_grids
import green_grids.repn.generator as generator

CHEBYSHEV_BASIS = "chebyshev"
IR_BASIS = "ir"


def add_common_parameters(parser: argparse.ArgumentParser):
    '''Common parameter definitions for all subparsers.

    @parser : ArgumentParser - parser to be updated
    '''
    parser.add_argument("--trim",
                        type=bool,
                        help="Trim number of coefficients if needed.",
                        default=True)
    parser.add_argument("--outfile",
                        type=str,
                        help="Name of the file to store generated data.",
                        required=True)


def add_ir_parameters(parser: argparse.ArgumentParser):
    '''
    Add IR specific parameters to CLI parser

    @parser : ArgumentParser - parser to be updated
    '''
    parser.add_argument("--irlambda",
                        type=str,
                        required=True,
                        help="Intermediate representation Lambda parameter.")
    parser.add_argument("--h5file",
                        type=str,
                        default="",
                        help="Existent HDF5 file with "
                             "Intermediate representation data")
    parser.add_argument("--ncoeff",
                        type=int,
                        help="Number of coefficients in the basis")


def ir_generator(args, stats):
    '''
    Construct IR generator for the specific statistics

    @args - command line arguments
    @stats - basis statistics, Fermi or Bose

    Returns
    -------
    generator : sparse_grid.repn.Generator
    '''
    return green_grids.get_generator(args.basis, args.irlambda, args.ncoeff,
                                     stats, args.trim, args.h5file)


def add_chebyshev_parameters(parser: argparse.ArgumentParser):
    '''
    Add Chebyshev specific parameters to CLI parser

    @parser : ArgumentParser - parser to be updated
    '''
    parser.add_argument("--ncoeff",
                        type=int,
                        help="Number of coefficients in the basis",
                        required=True)
    parser.add_argument(
        "--prec",
        type=float,
        default=None,
        help="Chebyshev basis desired precision in number of digits.")


def chebyshev_generator(args, stats):
    '''
    Construct Chebyshev generator for the specific statistics

    @args - command line arguments
    @stats - basis statistics, Fermi or Bose

    Returns
    -------
    generator : sparse_grid.repn.Generator
    '''
    return green_grids.get_generator(args.basis, args.ncoeff, stats, args.trim,
                                     args.prec)


BASIS_PARAMETERS = {
    IR_BASIS: {
        "params": add_ir_parameters,
        "generator": ir_generator
    },
    CHEBYSHEV_BASIS: {
        "params": add_chebyshev_parameters,
        "generator": chebyshev_generator
    }
}


def main():
    # Initialize command line arguments parser. We disable defualt help message
    # to generate basis specific help.
    parser = argparse.ArgumentParser(description="Sparse grid generator.")
    subparsers = parser.add_subparsers(
        dest='basis',
        required=True,
        title='Basis types',
        help='All available basis types.',
        description="Use 'generate.py <command> --help' to see"
                    " options for each basis type.")

    # Add a separate command for each basis type.
    for basis in BASIS_PARAMETERS:
        command = subparsers.add_parser(
            basis, help='Generate data for {} basis.'.format(basis))
        # Basis-specific parameter definitions.
        BASIS_PARAMETERS[basis]["params"](command)
        # Common parameter definitions.
        add_common_parameters(command)

    args = parser.parse_args()

    fermi_generator = BASIS_PARAMETERS[args.basis]["generator"](args, "fermi")
    bose_generator = BASIS_PARAMETERS[args.basis]["generator"](args, "bose")
    sparse_data = generator.generate_paired_sparse_data(
        fermi_generator, bose_generator, 'fermi', 'bose')
    sparse_data.save_hdf5(args.outfile)


if __name__ == "__main__":
    main()
