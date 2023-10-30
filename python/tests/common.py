import shutil
import tempfile
import unittest


class HDF5TestCase(unittest.TestCase):
    """Test case with temporary HDF5 file.
    """

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix='sparse_grid_test_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)

    def mktemp(self, suffix='.hdf5', prefix='', path=None):
        if path is None:
            path = self.tempdir
        return tempfile.mktemp(suffix, prefix, dir=path)
