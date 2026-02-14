import unittest

# import numpy as np
# from numpy.testing import assert_allclose

from green_grids import get_generator, get_fermi_bose_pair
from green_grids import get_transformer, transform_data, transform_paired_data


class ModuleFunctionsTest(unittest.TestCase):

    def setUp(self):
        self.beta = 5.
        gen = get_generator('chebyshev', 12, 'fermi')
        self.data = gen()
        self.pair = get_fermi_bose_pair('chebyshev', 12)

    def test_get_generator_invalid_repn(self):
        with self.assertRaises(KeyError):
            get_generator('invalid_type', 'blah', 'blah')

    def test_get_generator_chebyshev(self):
        self.assertIsNotNone(
            get_generator('chebyshev', stats='fermi', ncoeff=12))

    def test_get_generator_ir(self):
        self.assertIsNotNone(
            get_generator('ir', stats='bose', lambda_=1e2, ncoeff=None))

    def test_get_fermi_bose_pair_invalid_repn(self):
        with self.assertRaises(KeyError):
            get_fermi_bose_pair('invalid_type', 'blah', 'blah')

    def test_get_fermi_bose_pair_chebyshev(self):
        self.assertIsNotNone(get_fermi_bose_pair('chebyshev', 12))

    def test_get_fermi_bose_pair_ir(self):
        self.assertIsNotNone(get_fermi_bose_pair('ir', 1e2, None))

    def test_get_transformer_invalid_repn(self):
        with self.assertRaises(KeyError):
            get_transformer('invalid_type', 'blah')

    def test_get_transformer_valid_repn(self):
        self.assertIsNotNone(get_transformer('chebyshev', self.beta))
        self.assertIsNotNone(get_transformer('ir', self.beta))

    def test_transform_data_no_metadata(self):
        self.data.metadata = None
        # No metadata means no-op.
        self.assertIs(self.data, transform_data(self.data, self.beta))

    def test_transform_data_valid(self):
        self.assertIsNotNone(transform_data(self.data, self.beta))

        data, tranformer = transform_data(self.data,
                                          self.beta,
                                          return_transformer=True)
        self.assertIsNotNone(data)
        self.assertIsNotNone(tranformer)

    def test_transform_paired_data_no_metadata(self):
        self.pair.get_grid('fermi').metadata = None
        # No metadata means no-op.
        self.assertIs(self.pair, transform_paired_data(self.pair, self.beta))

    def test_transform_paired_data_valid(self):
        self.assertIsNotNone(transform_paired_data(self.pair, self.beta))
