import numpy as np
import unittest
from numpy.testing import assert_allclose
from sparse_grid.repn.chebyshev import Basis as ChebyshevBasis
from sparse_grid.repn.ir import Basis as IRBasis


class ChebyshevBasisTest(unittest.TestCase):

    def test_init(self):
        basis = ChebyshevBasis(32, 'fermi')
        self.assertIsNotNone(basis)

    def test_init_invalid_ncoeff(self):
        with self.assertRaises(AssertionError):
            ChebyshevBasis(0, 'fermi')

    def test_init_trim_and_warn(self):
        with self.assertWarns(RuntimeWarning):
            basis = ChebyshevBasis(31, 'fermi')

        self.assertEqual(30, basis.dim)

        with self.assertWarns(RuntimeWarning):
            basis = ChebyshevBasis(32, 'bose')

        self.assertEqual(31, basis.dim)

    def test_init_notrim_and_raise(self):
        with self.assertRaises(ValueError):
            ChebyshevBasis(31, 'fermi', trim=False)

        with self.assertRaises(ValueError):
            ChebyshevBasis(32, 'bose', trim=False)

    def test_fermi_grid(self):
        basis = ChebyshevBasis(2, 'fermi')
        xgrid_expected = np.array([np.sqrt(0.5), -np.sqrt(0.5)])
        wgrid_expected = np.array([-1, 0], dtype=int)

        assert_allclose(np.sort(basis.sampling_points_x()),
                        np.sort(xgrid_expected),
                        atol=1e-12)
        assert_allclose(basis.sampling_points_matsubara(), wgrid_expected)
        self.assertEqual(basis.stats, 'fermi')

    def test_bose_grid(self):
        basis = ChebyshevBasis(3, 'bose')
        self.assertEqual(basis.dim, 3)
        xgrid_expected = np.array([np.sqrt(0.75), 0., -np.sqrt(0.75)])
        wgrid_expected = np.array([-1, 0, 1], dtype=int)

        assert_allclose(np.sort(basis.sampling_points_x()),
                        np.sort(xgrid_expected),
                        atol=1e-12)
        assert_allclose(basis.sampling_points_matsubara(), wgrid_expected)
        self.assertEqual(basis.stats, 'bose')

    def test_bose_transform(self):
        uxl_data_0 = [
            1, 0.998716507171053, 0.994869323391895, 0.988468324328111,
            0.979529941252494, 0.968077118866203, 0.954139256400047,
            0.937752132147079, 0.918957811620228, 0.897804539570739
        ]
        uxl_data_10 = [
            1, 0.485301962531081, -0.528964010326963, -0.998716507171053,
            -0.440394151557634, 0.571268215094793, 0.994869323391895,
            0.394355855113318, -0.612105982547663, -0.988468324328111
        ]
        uwl_data_3 = [
            0, 0.0424413181578388j, 0.00360253097394979, 0.0419826296681221j,
            0.0143322545192895, 0.0386354384717829j, 0.0313368153970256,
            0.0278032563311303j, 0.051222475311555
        ]

        basis = ChebyshevBasis(31, 'bose')
        xgrid = basis.sampling_points_x()
        wgrid = basis.sampling_points_matsubara()
        uxl = basis.uxl(np.arange(xgrid.shape[0]), xgrid)
        unl = basis.compute_unl(wgrid)

        self.assertTrue(np.allclose(uxl[0, :10], uxl_data_0))
        self.assertTrue(np.allclose(uxl[10, :10], uxl_data_10))
        self.assertTrue(np.allclose(unl[3, :9], uwl_data_3))

    def test_fermi_transform(self):
        uxl_data_0 = [
            1, 0.998795456205172, 0.995184726672197, 0.989176509964781,
            0.980785280403231, 0.970031253194544, 0.956940335732209,
            0.941544065183021, 0.923879532511288, 0.903989293123445
        ]
        uxl_data_10 = [
            1, 0.514102744193222, -0.471396736825998, -0.998795456205172,
            -0.555570233019602, 0.427555093430283, 0.995184726672197,
            0.595699304492433, -0.382683432365091, -0.989176509964781
        ]
        uwl_data_3 = [
            -0.0385830165071261j, -0.000744324581394584, -0.0385255799319049j,
            -0.00669227300356164, -0.0374353110376173j, -0.0183756244578383,
            -0.0326075361721497j, -0.034532553985681, -0.0199567485901221j
        ]

        basis = ChebyshevBasis(32, 'fermi')
        xgrid = basis.sampling_points_x()
        wgrid = basis.sampling_points_matsubara()
        uxl = basis.uxl(np.arange(xgrid.shape[0]), xgrid)
        unl = basis.compute_unl(wgrid)

        self.assertTrue(np.allclose(uxl[0, :10], uxl_data_0))
        self.assertTrue(np.allclose(uxl[10, :10], uxl_data_10))
        self.assertTrue(np.allclose(unl[3, :9], uwl_data_3))


class IRBasisTest(unittest.TestCase):

    def test_init(self):
        basis = IRBasis(1e2, 32, 'fermi')
        self.assertIsNotNone(basis)

    def test_init_invalid_ncoeff(self):
        with self.assertRaises(AssertionError):
            IRBasis(1e2, 0, 'fermi')

    def test_init_trim_and_warn(self):
        with self.assertWarns(RuntimeWarning):
            basis = IRBasis(1e2, 31, 'fermi')

        self.assertEqual(30, basis.dim)

        with self.assertWarns(RuntimeWarning):
            basis = IRBasis(1e2, 32, 'bose')

        self.assertEqual(31, basis.dim)

    def test_init_notrim_and_raise(self):
        with self.assertRaises(ValueError):
            IRBasis(1e2, 31, 'fermi', trim=False)

        with self.assertRaises(ValueError):
            IRBasis(1e2, 32, 'bose', trim=False)

    def test_fermi_grid(self):
        basis = IRBasis(1e1, 2, 'fermi')
        xgrid_expected = np.array([-0.702569851992996, 0.702569851992996])
        wgrid_expected = np.array([-1, 0], dtype=int)

        assert_allclose(np.sort(basis.sampling_points_x()),
                        np.sort(xgrid_expected),
                        atol=1e-12)
        assert_allclose(basis.sampling_points_matsubara(), wgrid_expected)
        self.assertEqual(basis.stats, 'fermi')

    def test_bose_grid(self):
        basis = IRBasis(1e1, 3, 'bose')
        xgrid_expected = np.array([-0.838639551136778, 0., 0.838639551136778])
        wgrid_expected = np.array([-1, 0, 1], dtype=int)

        assert_allclose(np.sort(basis.sampling_points_x()),
                        np.sort(xgrid_expected),
                        atol=1e-12)
        assert_allclose(basis.sampling_points_matsubara(), wgrid_expected)
        self.assertEqual(basis.stats, 'bose')

    def test_bose_transform(self):
        uxl_data_0 = [
            13.745432400947653, -17.721221618995756,  21.429102790554534,
            -23.60428968283704 ,  22.3918778239321  , -21.95339837035436,
            19.861605527659655, -18.359544957426447,  16.36432753762062
        ]
        uxl_data_10 = [
            0.286823988021064, 0.094545173053576, -0.440994942887295,
            -0.208862534963973, 0.439843513911161,  0.331487269960403,
            -0.340586641751847, -0.436536359352603,  0.191329811128685
        ]
        uwl_data_8 = [
            0.255415222461193, 0.116395512483477j, -0.07675446293742,
            -0.232745340169882j, -0.197612226322009, 0.312355551945083j,
            0.321493009265652, -0.33201752882286j, -0.327777925931597
        ]

        basis = IRBasis(1e4, 19, 'bose')
        xgrid = basis.sampling_points_x()
        wgrid = basis.sampling_points_matsubara()
        uxl = basis.uxl(np.arange(xgrid.shape[0]), xgrid)
        unl = basis.compute_unl(wgrid)

        self.assertTrue(np.allclose(uxl[0, :9], uxl_data_0))
        self.assertTrue(np.allclose(uxl[10, :9], uxl_data_10))
        self.assertTrue(np.allclose(unl[8, :9], uwl_data_8))

    def test_fermi_transform(self):
        uxl_data_0 = [
            -12.054785431598697, 10.477147954856228, -8.981242780476586,
            7.532701884783, -6.146120687665874, 4.811408704447319,
            -3.531319270454278, 2.30281481655022, -1.126051446254826
        ]
        uxl_data_10 = [
            -0.322993474112357, 0.284162115673715, 0.372417191068422,
            -0.223040032390382, -0.412570125446417, 0.154215657870916,
            0.441635049846618, -0.079204602655054, -0.458106832919598,
        ]
        uwl_data_8 = [
            -0.179416049057764j, -0.222380421870855, 0.212669675007241j,
            0.189296146549642, -0.091963069860477j, -0.01417900648884,
            -0.078772018706229j, -0.152479503509744,  0.213106209267317j
        ]

        basis = IRBasis(1e4, 40, 'fermi')
        xgrid = basis.sampling_points_x(20)
        wgrid = basis.sampling_points_matsubara(20)
        uxl = basis.uxl(np.arange(xgrid.shape[0], dtype=int), xgrid)
        unl = basis.compute_unl(wgrid)

        self.assertTrue(np.allclose(uxl[0, -9:], uxl_data_0))
        self.assertTrue(np.allclose(uxl[10, -9:], uxl_data_10))
        self.assertTrue(np.allclose(unl[8, :9], uwl_data_8))

    def test_fermi_atomic_limit(self):
        basis = IRBasis(1e2, 10, 'fermi')
        xgrid = basis.sampling_points_x()
        wgrid = basis.sampling_points_matsubara()
        g_atomic_iw = 1./((2*wgrid + 1)*np.pi*1.j)
        uxl = basis.uxl(np.arange(xgrid.shape[0], dtype=int), xgrid)
        ulx = basis.ulx(uxl)
        unl = basis.compute_unl(wgrid)
        uln = basis.compute_uln(unl)
        g_atomic_x = np.einsum("xl,ln,n->x", uxl,uln,g_atomic_iw)
        g_atomic_iw_new = np.einsum("nl,lx,x->n", unl,ulx,g_atomic_x)
        self.assertTrue(np.allclose(g_atomic_iw, g_atomic_iw_new))

if __name__ == '__main__':
    unittest.main()
