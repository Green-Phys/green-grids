import pytest
import h5py
import numpy as np
from pathlib import Path
from green_grids import get_generator, generator


def gridfile_diff(file1, file2):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # Compare keys
        assert f1.attrs['__grids_version__'] == f2.attrs['__grids_version__'], "Grid file versions do not match."
        keys = ["ngrid", "u1l_neg", "u1l_pos", "ulw", "ulx", "uwl", "uxl", "uxl_other", "wgrid", "xgrid"]
        stats = ["fermi", "bose"]
        for st in stats:
            for key in keys:
                data1 = f1[f"{st}/{key}"][:]
                data2 = f2[f"{st}/{key}"][:]
                assert np.allclose(data1, data2, atol=1e-8), f"Difference found in {st}/{key}"


@pytest.mark.parametrize(
    "ir_lambda, outfile",
    [
        (1e4, "1e4.h5"),
        (1e5, "1e5.h5"),
        pytest.param(1e6, "1e6.h5", marks=pytest.mark.slow),
        pytest.param(1e7, "1e7.h5", marks=pytest.mark.slow),
        pytest.param(1e8, "1e8.h5", marks=pytest.mark.slow),
    ]
)
def test_ir_gridfiles(ir_lambda, outfile, tmp_path):
    # create temporary directory for generated files
    outfile = tmp_path / outfile

    # Syntax: get_generator(basis, ir_lambda, ncoeff, stats, trim, h5file to read)
    fermi_generator = get_generator("ir", ir_lambda, None, "fermi", True, None)
    bose_generator = get_generator("ir", ir_lambda, None, "bose", True, None)
    sparse_data = generator.generate_paired_sparse_data(fermi_generator, bose_generator, "fermi", "bose")
    # save data
    sparse_data.save_hdf5(outfile)

    # Compare HDF5 files
    test_dir = Path(__file__).resolve().parent
    repo_dir = test_dir.parent.parent
    data_dir = repo_dir / "data" / "ir"
    reference_file = data_dir / outfile

    # Check if the generated file matches the reference file
    assert reference_file.exists(), f"Reference file {reference_file} does not exist."
    assert Path(outfile).exists(), f"Generated file {outfile} does not exist."

    # Use the diff_hdf5 function to compare the contents of the HDF5 files
    gridfile_diff(reference_file, outfile)


@pytest.mark.parametrize(
    "ncoeff, outfile",
    [
        (100, "100.h5"),
        (150, "150.h5"),
        pytest.param(200, "200.h5", marks=pytest.mark.slow),
        pytest.param(300, "300.h5", marks=pytest.mark.slow),
        pytest.param(350, "350.h5", marks=pytest.mark.slow),
        pytest.param(450, "450.h5", marks=pytest.mark.slow),
    ]
)
def test_chebyshev_gridfiles(ncoeff, outfile, tmp_path):
    # create temporary directory for generated files
    outfile = tmp_path / outfile

    # Syntax: get_generator(basis, ncoeff, stats, trim, precision)
    fermi_generator = get_generator("chebyshev", ncoeff, "fermi", True, None)
    bose_generator = get_generator("chebyshev", ncoeff, "bose", True, None)
    sparse_data = generator.generate_paired_sparse_data(fermi_generator, bose_generator, "fermi", "bose")
    # save data
    sparse_data.save_hdf5(outfile)

    # Compare HDF5 files
    test_dir = Path(__file__).resolve().parent
    repo_dir = test_dir.parent.parent
    data_dir = repo_dir / "data" / "cheb"
    reference_file = data_dir / outfile

    # Check if the generated file matches the reference file
    assert reference_file.exists(), f"Reference file {reference_file} does not exist."
    assert Path(outfile).exists(), f"Generated file {outfile} does not exist."

    # Use the diff_hdf5 function to compare the contents of the HDF5 files
    gridfile_diff(reference_file, outfile)
