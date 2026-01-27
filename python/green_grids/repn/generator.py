import numpy as np
from green_grids.repn.basis import SparseBasis
from green_grids.sparse_data import SparseData, PairedSparseData
from warnings import warn

_stats_str = {'fermi': 'F', 'bose': 'B'}


class Generator:

    def __init__(self, basis: SparseBasis, ncoeff=None, trim=True):
        if ncoeff is None:
            ncoeff = basis.dim

        assert ncoeff is None or ncoeff > 0, 'Invalid number of coefficients (minimum is 1)'
        self.basis = basis

        if (ncoeff % 2 == 1) != (basis.stats == 'bose'):
            if trim:
                ncoeff -= 1
                warn(
                    "Bose (Fermi) must have even (odd) points. Using ncoeff={}."
                    .format(ncoeff), RuntimeWarning)
            else:
                raise ValueError("Bose (Fermi) must have even (odd) points.")

        self.stats = basis.stats
        self.lgrid = np.arange(ncoeff, dtype=int)
        self.ncoeff = ncoeff

    def __call__(self):
        ngrid = self.basis.sampling_points_matsubara(self.ncoeff)
        wgrid = (2. * ngrid + (1 if self.stats == 'fermi' else 0)) * np.pi
        xgrid = self.basis.sampling_points_x(self.ncoeff)

        uxl = self.eval_on_x(xgrid)
        ulx = self.basis.ulx(uxl)
        u1l_pos = self.basis.uxl(self.lgrid, 1.)
        u1l_neg = self.basis.uxl(self.lgrid, -1.)

        uwl = self.basis.compute_unl(ngrid, self.lgrid)
        ulw = self.basis.compute_uln(uwl)
        return SparseData(self.stats,
                          xgrid,
                          ngrid,
                          wgrid,
                          uxl,
                          ulx,
                          u1l_pos,
                          u1l_neg,
                          uwl,
                          ulw,
                          metadata=self.basis.metadata(self.ncoeff))

    def eval_on_x(self, xgrid):
        xgrid = np.asarray(xgrid)
        assert xgrid.ndim == 1
        return self.basis.uxl(self.lgrid, xgrid)

    def eval_on_w(self, ngrid):
        return self.basis.compute_unl(ngrid, self.lgrid)


def generate_paired_sparse_data(generator1: Generator,
                                generator2: Generator,
                                gridname1='grid1',
                                gridname2='grid2'):
    """Generate a PairedSparseData from two Generators.
    """
    grid1 = generator1()
    grid2 = generator2()

    return PairedSparseData(
        grid1,
        grid2,
        generator1.eval_on_x(grid2.xgrid),
        generator2.eval_on_x(grid1.xgrid),
        gridname1=gridname1,
        gridname2=gridname2,
    )
