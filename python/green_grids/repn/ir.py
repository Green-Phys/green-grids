import numpy as np
import scipy.linalg as la
from green_grids.repn.basis import SparseBasis
from green_grids.repn.transformer import SparseTransformer
from sparse_ir import poly as spir_poly
import sparse_ir

_stats_str = {'fermi': 'F', 'bose': 'B'}


class Basis(SparseBasis):

    def __init__(self, lambda_, ncoeff, stats, trim=True, h5file=""):
        spir_poly.PiecewiseLegendreFT._DEFAULT_GRID = np.hstack(
            [np.arange(2**6), (2**np.linspace(6, 30, 16 * (30 - 6) + 1)).astype(np.int64)]
        )
        self._basis = sparse_ir.FiniteTempBasis(_stats_str[stats], beta=1.0, wmax=int(float(lambda_)))
        self.Lambda = lambda_
        if ncoeff is None:
            ncoeff = self._basis.size
        super().__init__(ncoeff, stats, trim)

    def ulx(self, uxl):
        return la.pinv(uxl)

    def _sampling_points_matsubara(self, ncoeff: int):
        return (sparse_ir.basis._default_matsubara_sampling_points(self._basis._uhat_full, ncoeff) - (1 if self.stats == 'fermi' else 0)) // 2

    def _sampling_points_x(self, ncoeff: int):
        return sparse_ir.basis._default_sampling_points(self._basis.sve_result.u, ncoeff)

    def _uxl(self, l, x):
        return self._basis.u[:self._dim]((np.asarray(x) + 1) / 2).T * np.sqrt(1.0 / 2.0)

    def compute_unl(self, n, whichl=None):
        return self._basis.uhat[:self._dim]((2 * n + (1 if self.stats == 'fermi' else 0))).T

    def metadata(self, ncoeff):
        return {
            'type': 'ir',
            'ncoeff': ncoeff,
            'lambda': self.Lambda,
        }


def get_fermi_bose_basis_pair(lfermi: float,
                              nfermi: int,
                              trim=True,
                              lbose: float = None,
                              nbose: float = None,
                              h5file_fermi='',
                              h5file_bose=''):
    '''Get a pair of Basis with Fermi and Bose statistics, respectively.

    Args:
        lfermi: Lambda value for the Fermi basis.
        nfermi: Number of coefficients for the Fermi basis.
        trim: Bool. Whether to automatically trim incompatible ncoeffs.
        nfermi: Number of coefficients for the Fermi basis.
            Default value is lfermi.
        nbose: Number of coefficients for the Bose basis.
            Default value is nfermi (trimmed) minus 1.

    Returns:
        A tuple of Basis instances (fermi_basis, bose_basis).
    '''
    fermi_basis = Basis(lfermi,
                        nfermi,
                        'fermi',
                        trim=trim,
                        h5file=h5file_fermi)
    if not nbose:
        nbose = fermi_basis.dim - 1
    if not lbose:
        lbose = lfermi

    bose_basis = Basis(lbose, nbose, 'bose', trim=trim, h5file=h5file_bose)

    return (fermi_basis, bose_basis)


class Transformer(SparseTransformer):
    '''Implements `sparse_grid.repn.SparseTransformer`.
    '''

    def scale_fx(self, fx):
        return fx * np.sqrt(2 / self.beta)

    def scale_fx_inv(self, fx_inv):
        return fx_inv * np.sqrt(self.beta / 2)

    def scale_fiw(self, fiw):
        return fiw * np.sqrt(self.beta)

    def scale_fiw_inv(self, fiw_inv):
        return fiw_inv / np.sqrt(self.beta)

    @property
    def repn_type(self):
        return 'ir'
