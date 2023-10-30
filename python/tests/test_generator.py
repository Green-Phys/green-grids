import numpy as np
import unittest
from sparse_grid.repn.generator import Generator
from unittest.mock import Mock


def get_mock_basis(ncoeff, stats):
    mock_basis = Mock()
    mock_basis.dim = ncoeff
    mock_basis.stats = stats
    mock_basis.uxl = Mock(side_effect=[
        np.zeros([ncoeff, ncoeff]),
        np.zeros([
            ncoeff,
        ]),
        np.zeros([
            ncoeff,
        ])
    ])
    mock_basis.ulx = Mock(return_value=np.zeros([ncoeff, ncoeff]))
    mock_basis.compute_unl = Mock(return_value=np.zeros([ncoeff, ncoeff]))
    mock_basis.compute_uln = Mock(return_value=np.zeros([ncoeff, ncoeff]))
    mock_basis.sampling_points_x = Mock(return_value=np.zeros([ncoeff]))
    mock_basis.sampling_points_matsubara = Mock(
        return_value=np.zeros([ncoeff]))
    mock_basis.metadata = Mock(return_value={'type': 'mock', 'ncoeff': ncoeff})
    return mock_basis


class GeneratorTest(unittest.TestCase):

    def test_init(self):
        mock_basis = get_mock_basis(32, 'fermi')

        generator = Generator(mock_basis)
        self.assertIsNotNone(generator)

    def test_call_fermi(self):
        generator = Generator(get_mock_basis(2, 'fermi'))
        data = generator()

        self.assertEqual(data.stats, 'fermi')
        self.assertEqual(data.xgrid.shape, (2,))
        self.assertEqual(data.ngrid.shape, (2,))
        self.assertEqual(data.wgrid.shape, (2,))
        self.assertEqual(data.ulx.shape, (2, 2))
        self.assertEqual(data.uxl.shape, (2, 2))
        self.assertEqual(data.u1l_pos.shape, (2,))
        self.assertEqual(data.u1l_neg.shape, (2,))
        self.assertEqual(data.uwl.shape, (2, 2))
        self.assertEqual(data.ulw.shape, (2, 2))
        self.assertEqual(data.metadata['type'], 'mock')
        self.assertEqual(data.metadata['ncoeff'], 2)

    def test_call_bose(self):
        generator = Generator(get_mock_basis(3, 'bose'))
        data = generator()

        self.assertEqual(data.stats, 'bose')
        self.assertEqual(data.xgrid.shape, (3,))
        self.assertEqual(data.ngrid.shape, (3,))
        self.assertEqual(data.wgrid.shape, (3,))
        self.assertEqual(data.ulx.shape, (3, 3))
        self.assertEqual(data.uxl.shape, (3, 3))
        self.assertEqual(data.u1l_pos.shape, (3,))
        self.assertEqual(data.u1l_neg.shape, (3,))
        self.assertEqual(data.uwl.shape, (3, 3))
        self.assertEqual(data.ulw.shape, (3, 3))
        self.assertEqual(data.metadata['type'], 'mock')
        self.assertEqual(data.metadata['ncoeff'], 3)


if __name__ == '__main__':
    unittest.main()
