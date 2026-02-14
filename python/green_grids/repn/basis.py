import numpy as np
import scipy.linalg as la
from abc import ABC, abstractmethod
from warnings import warn


class SparseBasis(ABC):

    def __init__(self, ncoeff, stats, trim=True):
        assert stats in ('fermi', 'bose')
        self._trim = trim
        self._stats = stats
        self._dim = self.check_ncoeff(ncoeff)
        assert self._dim > 0

    def check_ncoeff(self, ncoeff):
        """
        Check that requested number of coefficients is correct for chosen statistics, even for Fermi
        and odd for Bose. If we use trimmed basis the number of coefficients will be adjusted and
        returned as the actual number of coefficients to be passed to the basis.

        Returns
        -------
        ncoeff : int
        """
        if (ncoeff % 2 == 1) != (self.stats == 'bose'):
            # trim basis dimension. Fermi basis should have an even size and Bose basis should have
            # an odd size
            if self._trim:
                ncoeff -= 1
                warn(
                    "Bose (Fermi) must have even (odd) points. Using ncoeff={}."
                    .format(ncoeff), RuntimeWarning)
            else:
                raise ValueError("Bose (Fermi) must have even (odd) points.")
        return ncoeff

    @property
    def dim(self):
        """
        Return dimension of basis

        Returns
        -------
        dim : int
        """
        return self._dim

    @property
    def stats(self):
        """
        Return grid statistics either Fermi or Bose
        """
        return self._stats

    @abstractmethod
    def _sampling_points_matsubara(self, ncoeff: int):
        """
        Internal implementation
        """

    def sampling_points_matsubara(self, ncoeff: int = None):
        """
        Return sampling points on the Matsubara grid of the requested grid size

        @ncoeff - size of the desired Matsubara grid

        Returns
        -------
        ngrid : ndarray(shape=(ncoeff,), dtype=int)
        """
        if ncoeff is None:
            ncoeff = self.dim
        ncoeff = self.check_ncoeff(ncoeff)
        return self._sampling_points_matsubara(ncoeff)

    @abstractmethod
    def _sampling_points_x(self, ncoeff: int):
        """
        Internal implementation
        """

    def sampling_points_x(self, ncoeff: int = None):
        """
        Return sampling points on the [-1; 1] grid of the requested grid size

        @ncoeff - size of the desired grid

        Returns
        -------
        xgrid : ndarray(shape=(ncoeff,), dtype=float)
        """
        if ncoeff is None:
            ncoeff = self.dim
        ncoeff = self.check_ncoeff(ncoeff)
        return self._sampling_points_x(ncoeff)

    @abstractmethod
    def _uxl(self, l, x):
        """
        Internal implementation of uxl function
        """

    def uxl(self, l, x):
        """
        Return the transformation matrix from the intermediate basis grid to the real-space grid

        @l - size of the intermediate grid
        @x - sampling points on the real-space grid

        Returns
        -------
        uxl : ndarray(shape=(x.shape[0],l), dtype=float)
        """
        xx = np.atleast_1d(x).ravel()
        ll = np.atleast_1d(l).ravel()
        res = self._uxl(ll, xx)
        return res[:, ll] if res.shape[0] > 1 else res[0, ll]

    @abstractmethod
    def ulx(self, uxl):
        """
        Return the transformation matrix from the real-space grid to the intermediate basis grid

        @uxl - transformation matrix from the intermediate basis grid to the real-space grid

        Returns
        -------
        ulx : ndarray(shape=(uxl.shape[1],uxl.shape[0]), dtype=float)
        """

    @abstractmethod
    def compute_unl(self, n, whichl=None):
        """
        Return the transformation matrix from the intermediate basis grid to Matsubara points grid

        @n - matsubara points grid

        Returns
        -------
        unl : ndarray(shape=(n.shape[0], n.shape[0]), dtype=float)
        """

    def compute_uln(self, unl):
        """
        Return the transformation matrix from the Matsubara points grid to the intermediate basis
        grid

        @unl - Transformation matrix from intermediate basis to Matsubara points

        Returns
        -------
        uln : ndarray(shape=(n.shape[0], n.shape[0]), dtype=float)
        """
        return la.pinv(unl)

    @abstractmethod
    def metadata(self, ncoeff):
        """
        Return the metadata for the basis of the desired size
        """
