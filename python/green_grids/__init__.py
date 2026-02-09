from .repn import chebyshev, ir, generator
from .sparse_data import SparseData, PairedSparseData
from .version import __version__


__all__ = ["__version__",]


ALL_REPNS = {
    'chebyshev': chebyshev,
    'ir': ir,
}
'''
Map of names and modules of all supported representations. Each module should
implement a `Generator` class and a `Transformer` class. See implementations for
chebyshev and ir for details
'''


def get_generator(repn_name: str, *args, **kwargs):
    '''
    Returns the `Generator` object corresponding to `repn_name`, forwarding
    `args` and `kwargs` to the constructor.
    '''

    if repn_name not in ALL_REPNS:
        raise KeyError('Unsupported repn {}'.format(repn_name))

    return generator.Generator(ALL_REPNS[repn_name].Basis(*args, **kwargs))


def get_fermi_bose_pair(repn_name: str, *args, **kwargs):
    '''Returns a `PairedSparseData` with a pair of Fermi and Bose grids.

    See `sparse_grid.repn.$repn_name.get_fermi_bose_basis_pair` for details.
    '''
    if repn_name not in ALL_REPNS:
        raise KeyError('Unsupported repn {}'.format(repn_name))

    fbasis, bbasis = ALL_REPNS[repn_name].get_fermi_bose_basis_pair(
        *args, **kwargs)
    return generator.generate_paired_sparse_data(generator.Generator(fbasis),
                                                 generator.Generator(bbasis),
                                                 gridname1='fermi',
                                                 gridname2='bose')


def get_transformer(repn_name: str, beta: float):
    '''
    Returns the `Transformer` object corresponding to `repn_name`, at inverse
    temperature `beta`.
    '''
    if repn_name not in ALL_REPNS:
        raise KeyError('Unsupported repn {}'.format(repn_name))

    return ALL_REPNS[repn_name].Transformer(beta)


def transform_data(data: SparseData,
                   beta: float,
                   return_transformer: bool = False):
    '''Transforms dimensionless `data` to a specific temperature.

    See `sparse_grid.repn.tranformer` for details.
    '''
    # No metadata means no-op.
    if not data.metadata:
        return (data, None) if return_transformer else data

    transformer = get_transformer(data.metadata['type'], beta)
    new_data = transformer.transform(data)
    return (new_data, transformer) if return_transformer else new_data


def transform_paired_data(paired_data: PairedSparseData, beta: float):
    '''Transforms dimensionless `data` to a specific temperature.

    See `sparse_grid.repn.tranformer` for details.
    '''
    names = list(paired_data.gridnames())
    for name in names:
        if not paired_data.get_grid(name).metadata:
            # No metadata means no-op.
            return paired_data

    new_pair = dict()
    for name in names:
        data = paired_data.get_grid(name)
        new_data, transformer = transform_data(data,
                                               beta,
                                               return_transformer=True)
        new_pair[name] = {
            'grid': new_data,
            'uxl_other': transformer.scale_fx(paired_data.get_uxl_other(name)),
        }

    grid1, grid2 = names
    return PairedSparseData(new_pair[grid1]['grid'],
                            new_pair[grid2]['grid'],
                            new_pair[grid1]['uxl_other'],
                            new_pair[grid2]['uxl_other'],
                            gridname1=grid1,
                            gridname2=grid2)
