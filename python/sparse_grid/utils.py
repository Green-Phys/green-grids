import h5py
import numpy as np
from warnings import warn


def h5save_dict(f, data: dict, basepath='/'):
    if not basepath.endswith('/'):
        basepath += '/'

    for k, v in data.items():
        if isinstance(v, dict):
            h5save_dict(f, v, basepath=basepath + k + '/')
        else:
            f[basepath + k] = v


def h5load_dict(h5obj):
    if isinstance(h5obj, h5py.Dataset):
        return h5obj[()]

    out = dict()
    for k, v in h5obj.items():
        out[k] = h5load_dict(v)
    return out


def checkreal(a, tol=1e-12):
    "Checks if quantity is real"
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.complexfloating):
        maxim = a.imag.max()
        minim = a.imag.min()
        imag_part = np.max([np.abs(maxim), np.abs(minim), maxim - minim])

        if imag_part > tol:
            warn("Imaginary part of magnitude %g" % imag_part, UserWarning, 2)
    return a.real
