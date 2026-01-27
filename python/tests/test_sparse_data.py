import h5py
import numpy as np
import os
import shutil
from contextlib import contextmanager
from green_grids.sparse_data import PairedSparseData, SparseData
from green_grids.utils import h5save_dict

from .common import HDF5TestCase


def get_default_data_config(nx: int, nw: int, nl: int, stats: str = 'bose'):
    return {
        'stats': stats,
        'xgrid': np.linspace(-1, 1, nx),
        'ngrid': np.arange(nw, dtype=int),
        'wgrid': np.pi * (2 * np.arange(nw) + (1 if stats == 'fermi' else 0)),
        'uxl': np.ones((nx, nl)),
        'ulx': np.ones((nl, nx)),
        'u1l_pos': np.ones(nl),
        'u1l_neg': np.ones(nl),
        'uwl': np.ones((nw, nl)),
        'ulw': np.ones((nl, nw))
    }


class SparseDataTest(HDF5TestCase):

    def setUp(self):
        self.nx = 3
        self.nw = 5
        self.nl = 2
        self.stats = 'bose'
        self.config = get_default_data_config(self.nx, self.nw, self.nl,
                                              self.stats)

    @contextmanager
    def get_test_data_filename(self, metadata=None):
        fname = self.mktemp()
        with h5py.File(fname, 'w') as h5file:
            for k, v in self.config.items():
                h5file[k] = v
            if metadata:
                h5save_dict(h5file, metadata, basepath='metadata')

        yield fname
        shutil.rmtree(fname, ignore_errors=True)

    def test_init(self):
        data = SparseData(**self.config)
        self.assertEqual(self.stats, data.stats)
        self.assertEqual(self.nx, data.nx)
        self.assertEqual(self.nl, data.nl)
        self.assertEqual(self.nw, data.nw)
        self.assertIsNone(data.metadata)

    def test_load_hdf5(self):
        with self.get_test_data_filename() as fname:
            data = SparseData.load_hdf5(fname)

        self.assertEqual(self.stats, data.stats)
        self.assertEqual(self.nx, data.nx)
        self.assertEqual(self.nl, data.nl)
        self.assertEqual(self.nw, data.nw)
        self.assertIsNone(data.metadata)

    def test_load_hdf5_with_metadata(self):
        with self.get_test_data_filename(metadata={'key': 'value'}) as fname:
            data = SparseData.load_hdf5(fname)

        self.assertEqual(self.stats, data.stats)
        self.assertEqual(self.nx, data.nx)
        self.assertEqual(self.nl, data.nl)
        self.assertEqual(self.nw, data.nw)
        self.assertIsNotNone(data.metadata)

    def test_save_hdf5(self):
        fname = self.mktemp()
        SparseData(**self.config).save_hdf5(fname)
        self.assertTrue(os.path.isfile(fname))

        data = SparseData.load_hdf5(fname)
        self.assertEqual(self.stats, data.stats)
        self.assertEqual(self.nx, data.nx)
        self.assertEqual(self.nl, data.nl)
        self.assertEqual(self.nw, data.nw)
        self.assertIsNone(data.metadata)

        shutil.rmtree(fname, ignore_errors=True)

    def test_save_hdf5_with_metadata(self):
        self.config['metadata'] = {'key': 'value'}
        fname = self.mktemp()
        SparseData(**self.config).save_hdf5(fname)
        self.assertTrue(os.path.isfile(fname))

        data = SparseData.load_hdf5(fname)
        self.assertEqual(self.stats, data.stats)
        self.assertEqual(self.nx, data.nx)
        self.assertEqual(self.nl, data.nl)
        self.assertEqual(self.nw, data.nw)
        self.assertIsNotNone(data.metadata)
        self.assertEqual(data.metadata['key'], b'value')

        shutil.rmtree(fname, ignore_errors=True)

    def test_init_invalid_stats(self):
        self.config['stats'] = 'invalid'
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_xgrid_dim(self):
        self.config['xgrid'] = np.ones((2, 3))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_ngrid_dim(self):
        self.config['ngrid'] = np.ones((2, 3), dtype=int)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_wgrid_dim(self):
        self.config['wgrid'] = np.ones((2, 3))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_uxl_shape(self):
        self.config['uxl'] = np.ones(3)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['uxl'] = np.ones((self.nx + 1, self.nl))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_ulx_shape(self):
        self.config['ulx'] = np.ones(3)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['ulx'] = np.ones((self.nl, self.nx + 1))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['ulx'] = np.ones((self.nl + 1, self.nx))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_u1l_pos_shape(self):
        self.config['u1l_pos'] = np.ones(self.nl + 1)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_u1l_neg_shape(self):
        self.config['u1l_neg'] = np.ones(self.nl + 1)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_uwl_shape(self):
        self.config['uwl'] = np.ones(3)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['uwl'] = np.ones((self.nw + 1, self.nl))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['uwl'] = np.ones((self.nw, self.nl + 1))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

    def test_init_invalid_ulw_shape(self):
        self.config['ulw'] = np.ones(3)
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['ulw'] = np.ones((self.nl + 1, self.nw))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)

        self.config['ulw'] = np.ones((self.nl, self.nw + 1))
        with self.assertRaises(AssertionError):
            SparseData(**self.config)


class PairedSparseDataTest(HDF5TestCase):

    def setUp(self):
        self.nx1 = 3
        self.nw1 = 5
        self.nl1 = 2
        self.stats1 = 'bose'
        self.gridname1 = 'grid1'
        self.config1 = get_default_data_config(self.nx1,
                                               self.nw1,
                                               self.nl1,
                                               stats=self.stats1)

        self.nx2 = 4
        self.nw2 = 7
        self.nl2 = 6
        self.stats2 = 'fermi'
        self.gridname2 = 'grid2'
        self.config2 = get_default_data_config(self.nx2,
                                               self.nw2,
                                               self.nl2,
                                               stats=self.stats2)

        self.uxl_other1 = np.ones((self.nx2, self.nl1))
        self.uxl_other2 = np.ones((self.nx1, self.nl2))

        self.args = None
        self.kwargs = None

    def setup_paired_config(self):
        self.args = [
            SparseData(**self.config1),  # grid1
            SparseData(**self.config2),  # grid2
            self.uxl_other1,
            self.uxl_other2,
        ]
        self.kwargs = {
            'gridname1': self.gridname1,
            'gridname2': self.gridname2,
        }

    @contextmanager
    def get_test_data_filename(self, metadata1=None, metadata2=None):
        self.setup_paired_config()

        fname = self.mktemp()
        with h5py.File(fname, 'w') as h5file:
            grp1 = h5file.require_group(self.gridname1)
            for k, v in self.config1.items():
                grp1[k] = v
            grp1['uxl_other'] = self.uxl_other1

            if metadata1:
                h5save_dict(grp1, metadata1, basepath='metadata')

            grp2 = h5file.require_group(self.gridname2)
            for k, v in self.config2.items():
                grp2[k] = v
            grp2['uxl_other'] = self.uxl_other2

            if metadata2:
                h5save_dict(grp2, metadata2, basepath='metadata')

        yield fname
        shutil.rmtree(fname, ignore_errors=True)

    def test_init(self):
        self.setup_paired_config()
        data = PairedSparseData(*self.args, **self.kwargs)
        self.assertTrue(self.gridname1 in data.gridnames())
        self.assertTrue(self.gridname2 in data.gridnames())

    def test_load_hdf5(self):
        with self.get_test_data_filename() as fname:
            data = PairedSparseData.load_hdf5(fname)
        self.assertTrue(self.gridname1 in data.gridnames())
        self.assertTrue(self.gridname2 in data.gridnames())

    def test_save_hdf5(self):
        self.setup_paired_config()
        fname = self.mktemp()
        PairedSparseData(*self.args, **self.kwargs).save_hdf5(fname)
        self.assertTrue(os.path.isfile(fname))

        data = PairedSparseData.load_hdf5(fname)
        self.assertTrue(self.gridname1 in data.gridnames())
        self.assertTrue(self.gridname2 in data.gridnames())

        shutil.rmtree(fname, ignore_errors=True)

    def test_init_invalid_name(self):
        self.gridname1 = "invalid name"
        self.setup_paired_config()
        with self.assertRaises(AssertionError):
            PairedSparseData(*self.args, **self.kwargs)

    def test_init_identical_names(self):
        self.gridname1 = "name"
        self.gridname2 = "name"
        self.setup_paired_config()
        with self.assertRaises(AssertionError):
            PairedSparseData(*self.args, **self.kwargs)

    def test_init_invalid_uxl_shape(self):
        self.uxl_other1 = np.ones((self.nx2 + 1, self.nl1))
        self.setup_paired_config()
        with self.assertRaises(AssertionError):
            PairedSparseData(*self.args, **self.kwargs)

        self.uxl_other1 = np.ones((self.nx2, self.nl1 + 1))
        self.setup_paired_config()
        with self.assertRaises(AssertionError):
            PairedSparseData(*self.args, **self.kwargs)
