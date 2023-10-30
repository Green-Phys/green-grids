import numpy as np
import scipy.linalg as la
from sparse_grid.repn.basis import SparseBasis
from sparse_grid.repn.transformer import SparseTransformer
from sparse_ir import adapter

_stats_str = {'fermi': 'F', 'bose': 'B'}


def _start_guesses(lambda_, n=1000):
    "Construct points on a logarithmically extended linear interval"
    x1 = np.arange(n)
    x2 = np.array(np.exp(np.linspace(np.log(n), np.log(lambda_ * 100), n)), dtype=int)
    x = np.unique(np.hstack((x1, x2)))
    return x


class Basis(SparseBasis):

    def __init__(self, lambda_, ncoeff, stats, trim=True, h5file=""):
        self._basis = adapter.load(_stats_str[stats], int(float(lambda_)))
        adapter._start_guesses = lambda n=1000: _start_guesses(self._basis.Lambda, n)
        if ncoeff is None:
            ncoeff = self._basis.dim()

        super().__init__(ncoeff, stats, trim)

    def ulx(self, uxl):
        return la.pinv(uxl)

    def _sampling_points_matsubara(self, ncoeff: int):
        return self._basis.sampling_points_matsubara(ncoeff - 1)

    def _sampling_points_x(self, ncoeff: int):
        return self._basis.sampling_points_x(ncoeff - 1)

    def _uxl(self, l, x):
        return self._basis.ulx(l[None, :], x[:, None])

    def compute_unl(self, n, whichl=None):
        return self._basis.compute_unl(n, whichl)

    def metadata(self, ncoeff):
        return {
            'type': 'ir',
            'ncoeff': ncoeff,
            'lambda': self._basis.Lambda,
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
