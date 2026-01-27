import h5py
import numpy as np
import shutil
from numpy.testing import assert_allclose
from green_grids import utils

from .common import HDF5TestCase


class UtilsTest(HDF5TestCase):

    def setUp(self):
        self.small_dict = {'key1': 'val1', 'key2': {'innerkey': 'innerval'}}

    def test_h5save_dict(self):
        fname = self.mktemp()
        with h5py.File(fname, 'w') as h5file:
            utils.h5save_dict(h5file, self.small_dict)

        with h5py.File(fname, 'r') as h5file:
            self.assertTrue('key1' in h5file)
            self.assertEqual(b'val1', h5file['key1'][()])
            self.assertTrue('key2' in h5file)
            self.assertTrue('innerkey' in h5file['key2'])
            self.assertEqual(b'innerval', h5file['key2']['innerkey'][()])

        shutil.rmtree(fname, ignore_errors=True)

    def test_h5load_dict(self):
        fname = self.mktemp()

        with h5py.File(fname, 'w') as h5file:
            h5file['key1'] = 'val1'
            grp = h5file.create_group('key2')
            grp['innerkey'] = 'innerval'

        with h5py.File(fname, 'r') as h5file:
            data = utils.h5load_dict(h5file)

        self.assertEqual(sorted(data.keys()), sorted(self.small_dict.keys()))
        self.assertEqual(b'val1', data['key1'])
        self.assertEqual(sorted(data['key2'].keys()),
                         sorted(self.small_dict['key2'].keys()))
        self.assertEqual(b'innerval', data['key2']['innerkey'])

    def test_check_real(self):
        real_list = np.array([1., 2., 3.])
        checked_real = utils.checkreal(real_list)
        assert_allclose(real_list, checked_real)

        small_imag_list = np.array([1. + 1e-13j, 2. + 1e-13j, 3. + 1e-13j])
        checked_real = utils.checkreal(small_imag_list)
        assert_allclose(small_imag_list.real, checked_real)

        big_imag_list = np.array([1. + 1e-1j, 2. + 1e-1j, 3. + 1e-1j])
        with self.assertWarns(UserWarning):
            checked_real = utils.checkreal(big_imag_list)
        assert_allclose(big_imag_list.real, checked_real)
