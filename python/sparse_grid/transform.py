import numpy as np

from sparse_grid.sparse_data import SparseData, PairedSparseData


def l_to_tau(data: SparseData, fl: np.ndarray):
    """Transforms data from the sparse coefficients to the tau grid.

    Args:
        data: SparseData instance.
        fl: Array to transform. First dimension should equal `data.nl`

    Returns:
        An array of the same dimension as `fl`. The first dimension equals
        `data.nx`, the rest are the same as `fl`.
    """
    assert fl.shape[0] == data.nl, \
        'First dimension should match {}'.format(data.nl)

    return np.einsum('ij,j...->i...', data.uxl, fl)


def l_to_tau_zero(data: SparseData, fl: np.ndarray):
    """Evaluates data from the sparse coefficients to `tau = 0` (`x = -1`).

    Args:
        data: SparseData instance.
        fl: Array to transform. First dimension should equal `data.nl`

    Returns:
        An array of shape `fl.shape[1:]`.
    """
    assert fl.shape[0] == data.nl, \
        'First dimension should match {}'.format(data.nl)

    return np.einsum('j,j...->...', data.u1l_neg, fl)


def l_to_tau_beta(data: SparseData, fl: np.ndarray):
    """Evaluates data from the sparse coefficients to `tau = beta` (`x = 1`).

    Args:
        data: SparseData instance.
        fl: Array to transform. First dimension should equal `data.nl`

    Returns:
        An array of shape `fl.shape[1:]`.
    """
    assert fl.shape[0] == data.nl, \
        'First dimension should match {}'.format(data.nl)

    return np.einsum('j,j...->...', data.u1l_pos, fl)


def l_to_other_tau(pair: PairedSparseData, grid_from: str, fl: np.ndarray):
    """Evaluates data from the sparse coefficients of `grid_from` to the tau
    grid of the other SparseData in `pair`.

    Args:
        data: PairedSparseData instance.
        grid_from: str. Name of the sparse grid to transform from.
        fl: Array to transform. First dimension should equal
            `pair.get_grid(grid_from).nl`.

    Returns:
        An array of the same dimension as `fl`. The first dimension equals
        `nx` of the other sparse grid in `pair`, the rest are the same as `fl`.
    """
    assert fl.shape[0] == pair.get_grid(grid_from).nl, \
        'First dimension should match {}'.format(pair.get_grid(grid_from).nl)

    return np.einsum('ij,j...->i...', pair.get_uxl_other(grid_from), fl)


def l_to_iw(data: SparseData, fl: np.ndarray):
    """Transforms data from the tau grid to the sparse coefficients.

    Args:
        data: SparseData instance.
        fl: Array to transform. First dimension should equal `data.nl`

    Returns:
        An array of the same dimension as `fl`. The first dimension equals
        `data.nl`, the rest are the same as `fl`.
    """
    assert fl.shape[0] == data.nl, \
        'First dimension should match {}'.format(data.nl)

    return np.einsum('ij,j...->i...', data.uwl, fl)


def tau_to_l(data: SparseData, ftau: np.ndarray):
    """Transforms data from the tau grid to the sparse coefficients.

    Args:
        data: SparseData instance.
        ftau: Array to transform. First dimension should equal `data.nx`

    Returns:
        An array of the same dimension as `ftau`. The first dimension equals
        `data.nl`, the rest are the same as `ftau`.
    """
    assert ftau.shape[0] == data.nx, \
        'First dimension should match {}'.format(data.nx)

    return np.einsum('ij,j...->i...', data.ulx, ftau)


def iw_to_l(data: SparseData, fiw: np.ndarray):
    """Transforms data from the sparse coefficients to the frequency grid.

    Args:
        data: SparseData instance.
        fiw: Array to transform. First dimension should equal `data.nw`

    Returns:
        An array of the same dimension as `fiw`. The first dimension equals
        `data.nl`, the rest are the same as `fiw`.
    """
    assert fiw.shape[0] == data.nw, \
        'First dimension should match {}'.format(data.nw)

    return np.einsum('ij,j...->i...', data.ulw, fiw)
