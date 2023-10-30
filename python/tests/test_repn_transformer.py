import numpy as np
import unittest
from numpy.testing import assert_allclose
from sparse_grid.repn.chebyshev import Transformer as ChebyshevTransformer
from sparse_grid.repn.ir import Transformer as IRTransformer
from sparse_grid.repn.transformer import SparseTransformer
from sparse_grid.sparse_data import SparseData


class FakeTransformer(SparseTransformer):
    """Transformer that does nothing except xgrid/wgrid transforms.
    """

    def scale_fx(self, fx):
        return fx

    def scale_fx_inv(self, fx_inv):
        return fx_inv

    def scale_fiw(self, fiw):
        return fiw

    def scale_fiw_inv(self, fiw_inv):
        return fiw_inv

    @property
    def repn_type(self):
        return 'fake'


def get_test_sparse_data(dim: int = 3, stats: str = 'fermi', metadata=None):
    """Returns a SparseData for testing purposes, in which actual values
    do not make any sense.
    """
    xgrid = np.linspace(-1, 1, dim)
    ngrid = np.arange(dim, dtype=int)
    wgrid = np.arange(dim, dtype=float)
    uxl = np.eye(dim) * 3.0
    ulx = np.eye(dim) / 3.0
    uwl = np.eye(dim) * 7.0
    ulw = np.eye(dim) / 7.0
    u1l_pos = np.ones(dim) * 2.0
    u1l_neg = np.ones(dim) * 5.0
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


class SparseTransformerTest(unittest.TestCase):

    def test_init(self):
        transformer = FakeTransformer(beta=17.)
        self.assertAlmostEqual(17., transformer.beta)
        self.assertEqual('fake', transformer.repn_type)

    def test_scale_xgrid(self):
        beta = 17.
        transformer = FakeTransformer(beta=beta)
        xgrid_transformed = transformer.scale_xgrid(
            np.array([-1.0, 0.0, 0.5, 1.0]))
        assert_allclose(np.array([0., beta * 0.5, beta * 0.75, beta]),
                        xgrid_transformed)

    def test_scale_wgrid(self):
        beta = 17.
        transformer = FakeTransformer(beta=beta)
        wgrid_transformed = transformer.scale_wgrid(np.array([0, 1, 2]))
        assert_allclose(np.array([0, 1, 2]) / beta, wgrid_transformed)

    def test_transform_raises_if_no_metadata(self):
        data = get_test_sparse_data(stats='bose')
        beta = 17.
        transformer = FakeTransformer(beta=beta)
        with self.assertRaises(AssertionError):
            transformer.transform(data)

    def test_transform_raises_if_type_mismatch(self):
        data = get_test_sparse_data(stats='fermi', metadata={'type': 'real'})
        beta = 17.
        transformer = FakeTransformer(beta=beta)
        with self.assertRaises(AssertionError):
            transformer.transform(data)

    def test_transform_warns_if_beta_mismatch(self):
        data = get_test_sparse_data(stats='fermi',
                                    metadata={
                                        'type': 'fake',
                                        'beta': 3.
                                    })
        beta = 17.
        transformer = FakeTransformer(beta=beta)
        with self.assertWarns(Warning):
            transformed = transformer.transform(data)
        self.assertIs(transformed, data)

    def test_transform_return_self_if_beta_matches(self):
        beta = 17.
        data = get_test_sparse_data(stats='fermi',
                                    metadata={
                                        'type': 'fake',
                                        'beta': beta
                                    })
        transformer = FakeTransformer(beta=beta)
        transformed = transformer.transform(data)
        self.assertIs(transformed, data)

    def test_transform_return_new_data_if_no_beta(self):
        data = get_test_sparse_data(stats='fermi', metadata={'type': 'fake'})
        beta = 17.
        transformer = FakeTransformer(beta=beta)
        transformed = transformer.transform(data)
        self.assertIsNot(transformed, data)


class ChebyshevTransformerTest(unittest.TestCase):

    def test_init(self):
        transformer = ChebyshevTransformer(beta=17.)
        self.assertAlmostEqual(17., transformer.beta)
        self.assertEqual('chebyshev', transformer.repn_type)

    def test_scale_fx(self):
        beta = 17.
        transformer = ChebyshevTransformer(beta=beta)
        test_data = np.random.rand(2, 3, 5)
        assert_allclose(test_data, transformer.scale_fx(test_data))
        assert_allclose(test_data, transformer.scale_fx_inv(test_data))

    def test_scale_fiw(self):
        beta = 17.
        transformer = ChebyshevTransformer(beta=beta)
        test_data = np.random.rand(2, 3, 5)
        assert_allclose(test_data * (beta / 2),
                        transformer.scale_fiw(test_data))
        assert_allclose(test_data / (beta / 2),
                        transformer.scale_fiw_inv(test_data))


class IRTransformerTest(unittest.TestCase):

    def test_init(self):
        transformer = IRTransformer(beta=17.)
        self.assertAlmostEqual(17., transformer.beta)
        self.assertEqual('ir', transformer.repn_type)

    def test_scale_fx(self):
        beta = 17.
        transformer = IRTransformer(beta=beta)
        test_data = np.random.rand(2, 3, 5)
        assert_allclose(test_data * np.sqrt(2 / beta),
                        transformer.scale_fx(test_data))
        assert_allclose(test_data / np.sqrt(2 / beta),
                        transformer.scale_fx_inv(test_data))

    def test_scale_fiw(self):
        beta = 17.
        transformer = IRTransformer(beta=beta)
        test_data = np.random.rand(2, 3, 5)
        assert_allclose(test_data * np.sqrt(beta),
                        transformer.scale_fiw(test_data))
        assert_allclose(test_data / np.sqrt(beta),
                        transformer.scale_fiw_inv(test_data))
