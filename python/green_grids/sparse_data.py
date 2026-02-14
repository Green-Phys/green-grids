import h5py
import numpy as np
import re
from dataclasses import dataclass
from .utils import h5save_dict, h5load_dict
from .version import __version__
from typing import Optional


@dataclass
class SparseData:
    """Holds necessary data for constructing sparse grids and transfromation
    matrices.
    """

    stats: str
    "['fermi'|'bose'] statistics."

    xgrid: np.ndarray
    "1-D array. Sparse grid in realspace, scaled to the interval [-1, 1]."

    ngrid: np.ndarray
    """1-D array. Sparse grid in Matsubara frequency space, stored as integer
    indices $n$ such that $\\omega_n = (2n+\\zeta)\\pi/\\beta$.
    """

    wgrid: np.ndarray
    """1-D array. Sparse grid in Matsubara frequency space. Default value
    corresponds to $\\beta=1$.
    """

    uxl: np.ndarray
    """2-D matrix. Transformation matrix from the basis coefficients `l` to the
    real space grid `x`, such that $f_x = U_{xl} f_l$.
    """

    ulx: np.ndarray
    """2-D matrix. Transformation matrix from the real space grid `x` to the
    basis coefficients `l`, such that $f_l = U_{lx} f_x$.
    """

    u1l_pos: np.ndarray
    """1-D array. Transformation vector from the basis coefficients `l` to the
    real space point where $x=1$.
    """

    u1l_neg: np.ndarray
    """1-D array. Transformation vector from the basis coefficients `l` to the
    real space point where $x=-1$.
    """

    uwl: np.ndarray
    """2-D matrix. Transformation matrix from the basis coefficients `l` to the
    frequency space grid `w`, such that $f_w = U_{wl} f_l$.
    """

    ulw: np.ndarray
    """2-D matrix. Transformation matrix from the frequency grid `w` to the basis
    coefficients `l`, such that $f_l = U_{lw} f_w$.
    """

    metadata: Optional[dict] = None
    "Optional metadata to store additional info."

    def __post_init__(self):
        "Post init check."
        self.check()

    @property
    def nx(self):
        "Number of real space grid points."
        return self.xgrid.size

    @property
    def nw(self):
        "Number of Matsubara grid points."
        return self.ngrid.size

    @property
    def nl(self):
        "Number of basis coefficients."
        return self.uxl.shape[1]

    def check(self):
        "Validate fields."
        assert self.stats in ('fermi', 'bose'), \
            'Invalid stats: {}'.format(self.stats)

        assert self.xgrid.ndim == 1, \
            'xgrid should have dimension 1, got {}'.format(self.xgrid.ndim)
        nx = self.xgrid.size

        assert self.ngrid.ndim == 1, \
            'ngrid should have dimension 1, got {}'.format(self.ngrid.ndim)
        nw = self.ngrid.size

        assert self.wgrid.shape == (nw,), \
            'shape of wgrid should be ({},)'.format(nw)

        assert self.uxl.ndim == 2, \
            'uxl should have dimension 2, got {}'.format(self.uxl.ndim)
        assert self.uxl.shape[0] == nx, \
            'first dimension of uxl should match length of xgrid'
        nl = self.uxl.shape[1]

        assert self.ulx.ndim == 2, \
            'ulx should have dimension 2, got {}'.format(self.ulx.ndim)
        assert self.ulx.shape == (nl, nx), \
            'shape of ulx should be ({}, {})'.format(nl, nx)

        assert self.u1l_pos.shape == (nl,), \
            'shape of u1l_pos should be ({},)'.format(nl)
        assert self.u1l_neg.shape == (nl,), \
            'shape of u1l_neg should be ({},)'.format(nl)

        assert self.uwl.ndim == 2, \
            'uwl should have dimension 2, got {}'.format(self.uwl.ndim)
        assert self.uwl.shape[0] == nw, \
            'first dimension of uwl should match length of wgrid'
        assert self.uwl.shape[1] == nl, \
            'first dimension of uwl should match that of uwl'

        assert self.ulw.ndim == 2, \
            'ulw should have dimension 2, got {}'.format(self.ulw.ndim)
        assert self.ulw.shape == (nl, nw), \
            'shape of ulw should be ({}, {})'.format(nl, nw)

    def save_to(self, h5grp: h5py.Group):
        "Save data to an HDF5 Group `h5grp`."
        h5grp['stats'] = self.stats
        h5grp['xgrid'] = self.xgrid
        h5grp['ngrid'] = self.ngrid
        h5grp['wgrid'] = self.wgrid
        h5grp['uxl'] = self.uxl
        h5grp['ulx'] = self.ulx
        h5grp['u1l_pos'] = self.u1l_pos
        h5grp['u1l_neg'] = self.u1l_neg
        h5grp['uwl'] = self.uwl
        h5grp['ulw'] = self.ulw
        if self.metadata:
            h5save_dict(h5grp, self.metadata, 'metadata')

    def save_hdf5(self, filename: str, basepath='/'):
        "Save data to an HDF5 file with `filename` under `basepath`."
        if not basepath.endswith('/'):
            basepath += '/'

        with h5py.File(filename, 'w') as f:
            grp = f.require_group(basepath)
            self.save_to(grp)

    @classmethod
    def load_from(cls, h5grp: h5py.Group):
        "Load data from an HDF5 Group `h5grp` and construct a new SparseData."
        stats = h5grp['stats'][()].decode('utf-8')
        xgrid = h5grp['xgrid'][()]
        ngrid = h5grp['ngrid'][()]
        wgrid = h5grp['wgrid'][()]
        uxl = h5grp['uxl'][()]
        ulx = h5grp['ulx'][()]
        u1l_pos = h5grp['u1l_pos'][()]
        u1l_neg = h5grp['u1l_neg'][()]
        uwl = h5grp['uwl'][()]
        ulw = h5grp['ulw'][()]
        metadata = None
        if 'metadata' in h5grp.keys():
            metadata = h5load_dict(h5grp['metadata'])

        return SparseData(stats,
                          xgrid,
                          ngrid,
                          wgrid,
                          uxl,
                          ulx,
                          u1l_pos,
                          u1l_neg,
                          uwl,
                          ulw,
                          metadata=metadata)

    @classmethod
    def load_hdf5(cls, filename: str, basepath='/'):
        """Load data from the HDF5 file with `filename` under `basepath`, and
        construct a new SparseData.
        """
        with h5py.File(filename, 'r') as f:
            grp = f[basepath]
            return cls.load_from(grp)


class PairedSparseData:
    """A pair of SparseData with transform matrices between each other.

    In addition to the two SparseData, this stores transformation matrices
    from the basis coefficients `l` of one SparseData to the real
    grid of the other SparseData. The transformation matrices are named
    `uxl_other`.
    """

    def __init__(self,
                 grid1: SparseData,
                 grid2: SparseData,
                 uxl_other1: np.ndarray,
                 uxl_other2: np.ndarray,
                 gridname1='grid1',
                 gridname2='grid2'):
        self.check_names(gridname1, gridname2)
        self._grids = {
            gridname1: grid1,
            gridname2: grid2,
        }

        self._uxl_other = {
            gridname1: uxl_other1,
            gridname2: uxl_other2,
        }

        self._othergrid = {
            gridname1: gridname2,
            gridname2: gridname1,
        }

        self.check()

    def get_grid(self, gridname: str):
        """Returns the SparseData instance corresponding to `gridname`.
        """
        return self._grids[gridname]

    def get_uxl_other(self, gridname: str):
        """Returns the transformation matrix from the basis coefficients of the
        SparseData given by `gridname` to the real space grid of the other
        SparseData.
        """
        return self._uxl_other[gridname]

    def gridnames(self):
        """Returns a generator of names of both grids.
        """
        return self._grids.keys()

    def check(self):
        """Validates shapes of transformation matrices.
        """
        for gridname in self.gridnames():
            grid = self.get_grid(gridname)
            other = self.get_grid(self._othergrid[gridname])
            assert self.get_uxl_other(gridname).shape == (other.nx, grid.nl), \
                'uxl_other should have shape {}'.format((other.nx, grid.nl))

    def check_names(self, gridname1, gridname2):
        """Checks the gridnames to make sure:

        1. They are not the same;
        2. They only include letters, digits, underscores, or dashes.
        """
        assert gridname1 != gridname2, 'Grid names should not be the same'

        assert re.match('^[A-Za-z0-9_-]*$', gridname1), \
            'Invalid grid name {}'.format(gridname1)
        assert re.match('^[A-Za-z0-9_-]*$', gridname2), \
            'Invalid grid name {}'.format(gridname2)

    def save_to(self, h5grp: h5py.Group):
        """Saves all data to an HDF5 group `h5grp`.

        This creates a subgroup for each SparseData instance, using the gridname
        as the group name. Contents of the SparseData as well as the
        corresponding transformation matrices will be saved under the subgroup.
        """
        for gridname in self.gridnames():
            grp = h5grp.require_group(gridname)
            self.get_grid(gridname).save_to(grp)
            grp['uxl_other'] = self.get_uxl_other(gridname)

    def save_hdf5(self, filename: str, basepath='/'):
        """Saves all data to an HDF5 file `filename` under `basepath`.

        See `save_to()` for details.
        """
        if not basepath.endswith('/'):
            basepath += '/'

        with h5py.File(filename, 'w') as f:
            if basepath == '/':
                f.attrs['__grids_version__'] = __version__
            grp = f.require_group(basepath)
            self.save_to(grp)

    @classmethod
    def load_from(cls, h5grp: h5py.Group, gridname1: str, gridname2: str):
        """Loads data from an HDF5 group `h5grp` with given grid names.

        This assumes that `h5grp` has subgroups named `gridname1` and
        `gridname2`. See `save_to()` for the HDF5 file format.
        """
        grp1 = h5grp[gridname1]
        grid1 = SparseData.load_from(grp1)
        uxl_other1 = grp1['uxl_other'][()]

        grp2 = h5grp[gridname2]
        grid2 = SparseData.load_from(grp2)
        uxl_other2 = grp2['uxl_other'][()]

        return PairedSparseData(grid1,
                                grid2,
                                uxl_other1,
                                uxl_other2,
                                gridname1=gridname1,
                                gridname2=gridname2)

    @classmethod
    def load_hdf5(cls, filename, basepath='/', gridnames=None):
        """Loads data from an HDF5 file `filename` under `basepath`.

        An optional tuple `gridnames` can be given to specify the gridnames.
        Otherwise they will be determined by assuming 1) there are only two
        subgroups under `basepath`, and 2) the names of the subgroups are used
        as gridnames.

        See `save_to()` for the HDF5 file format.
        """
        with h5py.File(filename, 'r') as f:
            grp = f[basepath]
            if not gridnames:
                gridnames = [k.split('/')[-1] for k in grp.keys()]
            assert len(gridnames) == 2, \
                'unable to determine grids in file {}'.format(filename)
            return cls.load_from(grp, gridnames[0], gridnames[1])
