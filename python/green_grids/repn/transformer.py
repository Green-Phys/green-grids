import numpy as np
from abc import ABC, abstractmethod
from green_grids.sparse_data import SparseData
from warnings import warn


class SparseTransformer(ABC):
    """Abstract base class for Transformer.

    Each repn should implement this to correctly scale dimensionless data to
    include temperature info. Specifically:

    - `scale_wgrid()` and `scale_xgrid()`: provided by base class.
    - `scale_fx()` should be implemented to scale `uxl`, `u1l_pos`, `u1l_neg`,
      `uxl_other`, ...
    - `scale_fx_inv()` should be implemented to scale `ulx`.
    - `scale_fiw()` should be implemented to scale `uwl`, `uwl_other`.
    - `scale_fiw_inv()` should be implemented to scale `ulw`.
    - `repn_name()` should be implemented to return correct name string (e.g.
      "chebyshev" or "ir").

    See method `transform()` for details.
    """

    def __init__(self, beta: float):
        self._beta = beta

    def scale_xgrid(self, x):
        """Scale the real space grid `x` linearly from `[-1, 1]` to `[0, beta]`.
        """
        return (x + 1.0) * (self.beta / 2.0)

    def scale_wgrid(self, w):
        """Scale the matsubara grid `w` to the specific temperature.
        """
        return w / self.beta

    @abstractmethod
    def scale_fx(self, fx):
        """Scale function values in real space from f(x) to f(tau). Useful for
        scaling matrix elements uxl, u1l_pos, u1l_neg, ...
        """

    @abstractmethod
    def scale_fx_inv(self, fx_inv):
        """Scale function values in real space from f(tau) to f(x). Useful for
        scaling matrix elements ulx.
        """

    @abstractmethod
    def scale_fiw(self, fiw):
        """Scale function values in frequency space for f(iw). The difference
        is usually due to a Fourier prefactor. Useful for scaling matrix
        elements uwl, ...
        """

    @abstractmethod
    def scale_fiw_inv(self, fiw_inv):
        """Inverse scaling of `scale_fiw()`. Useful for scaling matrix elements
        ulw, ...
        """

    @property
    def beta(self):
        "Inverse temperature."
        return self._beta

    @property
    @abstractmethod
    def repn_type(self):
        """Type of the repn, e.g. "ir" or "chebyshev".
        """

    def transform(self, data: SparseData):
        """Transformation for `data`. Tags data by appending entries in
        `data.metadata` with the following info:
        - `beta`: `float`, inverse temperature.

        If transformation is successful, it will return a new SparseData
        instance.
        """
        assert data.metadata, 'Cannot transform data with no metadata.'
        assert data.metadata['type'] == self.repn_type, \
            'Cannot transform {} data using {} transformer'.format(
                data.metadata['type'], self.repn_type)

        # Returns `data` if `beta` is already in metadata.
        if 'beta' in data.metadata:
            if not np.isclose(self.beta, data.metadata['beta']):
                warn(
                    'Failed to transform data to beta={}. Already at beta={}.'.
                    format(self.beta, data.metadata['beta']))
            return data

        xgrid = self.scale_xgrid(data.xgrid)
        ngrid = data.ngrid
        wgrid = self.scale_wgrid(data.wgrid)
        uxl = self.scale_fx(data.uxl)
        u1l_pos = self.scale_fx(data.u1l_pos)
        u1l_neg = self.scale_fx(data.u1l_neg)
        ulx = self.scale_fx_inv(data.ulx)
        uwl = self.scale_fiw(data.uwl)
        ulw = self.scale_fiw_inv(data.ulw)
        metadata = data.metadata.copy()
        metadata['beta'] = self.beta

        return SparseData(data.stats,
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
