import numpy as np
import sparse_grid.transform as tf
import unittest
from numpy.testing import assert_allclose
from sparse_grid import get_fermi_bose_pair


class TransformTest(unittest.TestCase):

    def setUp(self):
        self.dim = 10
        self.pair = get_fermi_bose_pair('chebyshev', self.dim)

        def T0_x(t):
            return np.ones_like(t)

        def T1_x(x):
            return x

        def T0_w_fermi(w):
            return 4.j / w

        def T0_w_bose(w):
            return 2. if w == 0 else 0.

        def T1_w_fermi(w):
            return -8 / (w * w)

        def T1_w_bose(w):
            return 0. if w == 0 else -4.j / w

        # A set of precalculated values
        self.fl = np.zeros(self.dim)
        self.fl[0] = 2
        self.fl[1] = 3

        self.fzero = 2. - 3.
        self.fbeta = 2. + 3.

        self.fx_fermi = np.array([
            2. * T0_x(x) + 3. * T1_x(x)
            for x in self.pair.get_grid('fermi').xgrid
        ])
        self.fx_bose = np.array([
            2. * T0_x(x) + 3. * T1_x(x)
            for x in self.pair.get_grid('bose').xgrid
        ])
        self.fw_fermi = np.array([
            2. * T0_w_fermi(w) + 3. * T1_w_fermi(w)
            for w in self.pair.get_grid('fermi').wgrid
        ])
        self.fw_bose = np.array([
            2. * T0_w_bose(w) + 3. * T1_w_bose(w)
            for w in self.pair.get_grid('bose').wgrid
        ])

    @property
    def grid_fermi(self):
        return self.pair.get_grid('fermi')

    @property
    def grid_bose(self):
        return self.pair.get_grid('bose')

    def test_l_to_tau(self):
        with self.assertRaises(AssertionError):
            tf.l_to_tau(self.grid_fermi, np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.l_to_tau(self.grid_bose, np.empty(self.dim))

        assert_allclose(self.fx_fermi, tf.l_to_tau(self.grid_fermi, self.fl))
        assert_allclose(self.fx_bose, tf.l_to_tau(self.grid_bose,
                                                  self.fl[:-1]))

    def test_l_to_tau_zero(self):
        with self.assertRaises(AssertionError):
            tf.l_to_tau_zero(self.grid_fermi, np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.l_to_tau_zero(self.grid_bose, np.empty(self.dim))

        assert_allclose(self.fzero, tf.l_to_tau_zero(self.grid_fermi, self.fl))
        assert_allclose(self.fzero,
                        tf.l_to_tau_zero(self.grid_bose, self.fl[:-1]))

    def test_l_to_tau_beta(self):
        with self.assertRaises(AssertionError):
            tf.l_to_tau_beta(self.grid_fermi, np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.l_to_tau_beta(self.grid_bose, np.empty(self.dim))

        assert_allclose(self.fbeta, tf.l_to_tau_beta(self.grid_fermi, self.fl))
        assert_allclose(self.fbeta,
                        tf.l_to_tau_beta(self.grid_bose, self.fl[:-1]))

    def test_tau_to_l(self):
        with self.assertRaises(AssertionError):
            tf.tau_to_l(self.grid_fermi, np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.tau_to_l(self.grid_bose, np.empty(self.dim))

        assert_allclose(self.fl,
                        tf.tau_to_l(self.grid_fermi, self.fx_fermi),
                        atol=1e-12)
        assert_allclose(self.fl[:-1],
                        tf.tau_to_l(self.grid_bose, self.fx_bose),
                        atol=1e-12)

    def test_l_to_other_tau(self):
        with self.assertRaises(AssertionError):
            tf.l_to_other_tau(self.pair, 'fermi', np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.l_to_other_tau(self.pair, 'bose', np.empty(self.dim))

        assert_allclose(self.fx_bose,
                        tf.l_to_other_tau(self.pair, 'fermi', self.fl))
        assert_allclose(self.fx_fermi,
                        tf.l_to_other_tau(self.pair, 'bose', self.fl[:-1]))

    def test_l_to_iw(self):
        with self.assertRaises(AssertionError):
            tf.l_to_iw(self.grid_fermi, np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.l_to_iw(self.grid_bose, np.empty(self.dim))

        assert_allclose(self.fw_fermi, tf.l_to_iw(self.grid_fermi, self.fl))
        assert_allclose(self.fw_bose, tf.l_to_iw(self.grid_bose, self.fl[:-1]))

    def test_iw_to_l(self):
        with self.assertRaises(AssertionError):
            tf.iw_to_l(self.grid_fermi, np.empty(self.dim + 1))
        with self.assertRaises(AssertionError):
            tf.iw_to_l(self.grid_bose, np.empty(self.dim))

        assert_allclose(self.fl,
                        tf.iw_to_l(self.grid_fermi, self.fw_fermi),
                        atol=1e-12)
        assert_allclose(self.fl[:-1],
                        tf.iw_to_l(self.grid_bose, self.fw_bose),
                        atol=1e-12)
