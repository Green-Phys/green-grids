import numpy as np
import unittest
from numpy.testing import assert_allclose
from green_grids.repn.chebyshev import Basis as ChebyshevBasis
from green_grids.repn.ir import Basis as IRBasis


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
        xgrid_expected = np.array([0.5, -0.5])
        wgrid_expected = np.array([-1, 0], dtype=int)

        assert_allclose(np.sort(basis.sampling_points_x()),
                        np.sort(xgrid_expected),
                        atol=1e-12)
        assert_allclose(basis.sampling_points_matsubara(), wgrid_expected)
        self.assertEqual(basis.stats, 'fermi')

    def test_bose_grid(self):
        basis = IRBasis(1e1, 3, 'bose')
        xgrid_expected = np.array([-8.971930e-01, 0.0, 8.971930e-01])
        wgrid_expected = np.array([-1, 0, 1], dtype=int)

        assert_allclose(np.sort(basis.sampling_points_x()),
                        np.sort(xgrid_expected),
                        atol=1e-12)
        assert_allclose(basis.sampling_points_matsubara(), wgrid_expected)
        self.assertEqual(basis.stats, 'bose')

    def test_bose_transform(self):
        uxl_data_0 = [
            53.0330395953343, -53.0330396010825, 35.2351082797355,
            -35.2351085579774, 27.7300102274779, -27.7300150463786,
            23.1488795665343, -23.1489313987664, 19.9105082263949
        ]
        uxl_data_10 = [
            1.67201621220835e-05, 1.6714905012999e-05, -0.000593484301295908,
            -0.000593284323434273, 0.00718234433152259, 0.00717953428056469,
            -0.0489376157529712, -0.0489124370514369, 0.217033451984615
        ]
        uwl_data_8 = [
            0.0267773343787626, 0.00011984631703555j, -0.0444515093312776,
            -0.00100295468396756j, 0.0676847929082761, 0.00447541434393243j,
            -0.0953634348544582, -0.0147125670905463j, 0.122735156218375
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
            -19.6268475731047, 18.6229921151091, -17.6693953629138,
            16.7182220035211, -15.7919670482309, 14.8754781258815,
            -13.9748611946433, 13.0855267310788, -12.2091799590797
        ]
        uxl_data_10 = [
            -0.309710530627941, 0.297276179691631, 0.358903419986585,
            -0.241314482851883, -0.400131026972182, 0.177890877835628,
            0.431725549355425, -0.108196471684818, -0.452262578237676
        ]
        uwl_data_8 = [
            -0.179416049057764j, -0.222380421870855, 0.212669675007241j,
            0.189296146549642, -0.0919630698604766j, -0.0141790064888402,
            -0.0787720187062291j, -0.152479503509744, 0.213106209267317j
        ]

        basis = IRBasis(1e4, 40, 'fermi')
        xgrid = basis.sampling_points_x(20)
        wgrid = basis.sampling_points_matsubara(20)
        uxl = basis.uxl(np.arange(xgrid.shape[0], dtype=int), xgrid)
        unl = basis.compute_unl(wgrid)

        self.assertTrue(np.allclose(uxl[0, -9:], uxl_data_0))
        self.assertTrue(np.allclose(uxl[10, -9:], uxl_data_10))
        self.assertTrue(np.allclose(unl[8, :9], uwl_data_8))


if __name__ == '__main__':
    unittest.main()
