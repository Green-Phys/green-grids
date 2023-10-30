# Adapted from code by Markus Wallerberger from CQMP/pygf2.

import mpmath as mp
import numpy as np

from sparse_grid import utils
from sparse_grid.repn.basis import SparseBasis
from sparse_grid.repn.transformer import SparseTransformer


def get_xroot(order):
    "Roots for order'th polynomial"
    return np.cos(np.pi / order * (np.arange(order) + .5))


def get_fittau_matrix(order):
    "Fitting matrix from roots to Chebyshev coefficients"
    n = np.arange(order)[:, None]
    k = np.arange(order)[None, :]
    coeff = 2. / order * np.cos(np.pi / order * n * (k + 0.5))
    coeff[0, :] /= 2
    return coeff


def get_evaltau_matrix(order):
    "Evaluation matrix from Chebyshev coefficients to roots"
    x = get_xroot(order)
    return np.polynomial.chebyshev.chebvander(x, order - 1)


def gen_Imz(mmax, stat='fermi'):
    """Generates transformation polynomials from Chebyshev to Matsubara"""
    Ioddsum = np.zeros(mmax + 1, object)
    Ievensum = np.zeros(mmax + 1, object)
    Icurr = np.zeros(mmax + 1, object)

    if stat == 'fermi':
        # Initialize the sum to half of the first term allows a compact
        # statement of the recursion.
        Icurr[1] = 2
        Ievensum[1] = 1
        sign = -1
    elif stat == 'bose':
        sign = +1
    else:
        raise ValueError("statistics must be either fermi or bose, but is %s" %
                         stat)

    # First term
    yield Icurr.copy()
    for m in range(1, mmax):
        if m % 2:
            # Odd term
            Icurr[0] = 0
            Icurr[1:] = 2 * m * Ievensum[:-1]
            Icurr[1] += -1 - sign
            Ioddsum += Icurr
        else:
            # Even term
            Icurr[0] = 0
            Icurr[1:] = 2 * m * Ioddsum[:-1]
            Icurr[1] += 1 - sign
            Ievensum += Icurr
        yield Icurr.copy()


def cheb_integrate(n):
    """Evaluate the integral over [-1,1] of the n-th Chebyshev"""

    def _general(n):
        return ((-1.) ** n + 1) / (1 - n ** 2)

    n = np.asarray(n)
    result = np.zeros(n.shape, float)
    n_is_one = n == 1
    result[n_is_one] = 0
    result[~n_is_one] = _general(n[~n_is_one])
    return result


def eval_Imz(poly, stat, n, prec):
    """Evaluate transformation polys for desired frequency"""
    if stat == 'fermi':
        with mp.workdps(prec):
            z = 1 / (-1j * (n + 0.5) * mp.pi)
            return complex(mp.polyval(tuple(poly[::-1]), z))
    elif stat == 'bose':
        # The case n == 0 has to be handled specially, as the polynomial has
        # a pole there.
        if n == 0:
            order = poly.nonzero()[0].max(initial=0)
            return cheb_integrate(order)

        with mp.workdps(prec):
            z = 1 / (-1j * n * mp.pi)
            return complex(mp.polyval(tuple(poly[::-1]), z))
    else:
        raise ValueError("stat must be either fermi or bose")


def _sign_changes(fn):
    "Given a discretized 1D function, return the location of the extrema"
    fn = np.asarray(fn)
    sign_flip = fn[1:] * fn[:-1] < 0
    return sign_flip.nonzero()[0]


def iwroots(poly, stat, prec):
    """Get points close to the roots of Chebyshev on the Matsubara axis"""
    if stat not in ('fermi', 'bose'):
        raise ValueError("Illegal value of stat: %s" % stat)
    l = poly.nonzero()[0].max(initial=0)
    nw_bound = max(l, (l // 2) ** 2)  # Empirical bound
    fn = np.asarray(
        [eval_Imz(poly, stat, n, prec) for n in range(nw_bound + 1)])

    # Even functions are purely real, odd functions are purely imaginary
    if l % 2:
        fn = fn.imag
    else:
        fn = utils.checkreal(fn)
    n0 = _sign_changes(fn)

    # The first roots are close to, but just before the first Matsubaras, so
    # they are invisible at the Matsubara points
    if n0.size:
        nfirst = np.arange(n0.min())
        n0 = np.hstack((nfirst, n0))

    if stat == 'fermi':
        n0 = np.hstack((-n0[::-1] - 1, n0))
    else:
        n0 = n0[1:]  # remove the zero
        n0 = np.hstack((-n0[::-1], 0, n0))

    return n0


def get_matstf(order, stat, prec=None):
    """Transform matrix Chebyshev coefficients to Matsubara roots"""
    if stat not in ('fermi', 'bose'):
        raise ValueError("Illegal value of stat: %s" % stat)
    if (order % 2 == 1) != (stat == 'bose'):
        raise ValueError("Bose (Fermi) must have even (odd) points")
    if prec is None:
        prec = 2.5 * order

    poly = list(gen_Imz(order + 1, stat))
    roots = iwroots(poly[order], stat, prec)
    tfmat = np.reshape([
        eval_Imz(poly_i, stat, n_i, prec) for n_i in roots
        for poly_i in poly[:-1]
    ], (order, order))
    return roots, tfmat


class Basis(SparseBasis):

    def __init__(self, ncoeff, stats, trim=True, prec=None):
        super().__init__(ncoeff, stats, trim)
        self.__prec = 2.5 * self.dim if prec is None else prec
        self._poly = list(gen_Imz(self.dim + 1, stats))

    def _sampling_points_matsubara(self, ncoeff: int):
        return iwroots(self._poly[ncoeff], self.stats, self.__prec)

    def _sampling_points_x(self, ncoeff: int):
        return get_xroot(ncoeff)

    def _uxl(self, l, x):
        return np.polynomial.chebyshev.chebvander(x, np.max(l))

    def ulx(self, uxl):
        return get_fittau_matrix(uxl.shape[0])

    def compute_unl(self, n, whichl=None):
        wgrid = np.asarray(n)
        return np.reshape([
            eval_Imz(poly_i, self.stats, n_i, self.__prec) for n_i in wgrid
            for poly_i in self._poly[:-1]
        ], (wgrid.size, whichl.shape[0] if whichl is not None else self.dim))

    def metadata(self, ncoeff):
        return {'type': 'chebyshev', 'ncoeff': ncoeff}


def get_fermi_bose_basis_pair(nfermi: int, trim=True, prec=None, nbose=None):
    '''Get a pair of Basis with Fermi and Bose statistics, respectively.

    Args:
        nfermi: Number of coefficients for the Fermi basis.
        trim: Bool. Whether to automatically trim incompatible ncoeffs.
        prec: Working precision for mpmath.
        nbose: Number of coefficients for the Bose basis.
            Default value is nfermi (trimmed) minus 1.

    Returns:
        A tuple of Basis instances (fermi_basis, bose_basis).
    '''
    fermi_basis = Basis(nfermi, 'fermi', trim=trim, prec=prec)
    if not nbose:
        nbose = fermi_basis.dim - 1

    bose_basis = Basis(nbose, 'bose', trim=trim, prec=prec)

    return (fermi_basis, bose_basis)


class Transformer(SparseTransformer):
    '''Implements `sparse_grid.repn.SparseTransformer`.
    '''

    def scale_fx(self, fx):
        return fx

    def scale_fx_inv(self, fx_inv):
        return fx_inv

    def scale_fiw(self, fiw):
        return fiw * (self.beta / 2)

    def scale_fiw_inv(self, fiw_inv):
        return fiw_inv * (2 / self.beta)

    @property
    def repn_type(self):
        return 'chebyshev'
