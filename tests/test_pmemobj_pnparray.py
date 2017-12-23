# -*- coding: utf8 -*-
import unittest
from nvm import pmemobj
from tests.support import TestCase

numpy_avail = True
try:
    import numpy as np
except ImportError:
    numpy_avail = False

if numpy_avail:
    class TestPersistentNPArray(TestCase):

        def _make_nparray(self, *arg, **kw):
            self.fn = self._test_fn()
            self.pop = pmemobj.create(
                        self.fn, pool_size=64 * 1024 * 1024, debug=True)
            self.addCleanup(self.pop.close)
            self.pop.root = self.pop.new(pmemobj.PersistentNPArray, *arg, **kw)
            return self.pop.root

        def _reread_nparray(self):
            self.pop.close()
            self.pop = pmemobj.open(self.fn)
            return self.pop.root

        def test_creation_args(self):
            for param in [None, [1, 2, 3, 4, 5]]:
                with self.assertRaises(TypeError):
                    self._make_nparray(param)
                with self.assertRaises(TypeError):
                    self._make_nparray(param, dtype="int16", shape=1)
                with self.assertRaises(ValueError):
                    self._make_nparray(param,
                                       dtype=np.dtype(np.int16), shape=-1)
                with self.assertRaises(TypeError):
                    self._make_nparray(param,
                                       dtype=np.dtype(np.int16), shape=True)
                with self.assertRaises(ValueError):
                    self._make_nparray(param,
                                       dtype=np.dtype(np.int16), shape=(1, 2))
                with self.assertRaises(ValueError):
                    self._make_nparray(param,
                                       dtype=np.dtype(np.int16), shape=())

        def test_eq(self):
            array_a = self._make_nparray(range(1024 * 1024),
                                         dtype=np.dtype(np.int64))
            array_b = np.array(range(1024 * 1024))
            self.assertEqual(array_a.array.all(), array_b.all())

        def test_persist(self):
            array_a = self._make_nparray(range(1024 * 1024),
                                         dtype=np.dtype(np.int64))
            array_b = np.array(range(1024 * 1024))
            array_a = self._reread_nparray()
            self.assertEqual(str(array_a.array.dtype), "int64")
            self.assertEqual(array_a.array.shape, array_b.shape)
            self.assertEqual(array_a.array.all(), array_b.all())

        def test_index_and_slices(self):
            array_a = self._make_nparray(range(1000),
                                         dtype=np.dtype(np.int64))
            array_b = np.array(range(1000))
            for count in range(2):
                self.assertEqual(array_a[0], array_b[0])
                for idx in range(0, 1000):
                    self.assertEqual(array_a[idx], array_b[idx])
                    self.assertEqual(array_a[-idx], array_b[-idx])
                for idx in range(1, 1000, 10):
                    self.assertEqual(array_a[idx:idx + 10].all(),
                                     array_b[idx:idx + 10].all())
                    self.assertEqual(array_a[:idx].all(),
                                     array_b[:idx].all())
                    self.assertEqual(array_a[:-idx].all(),
                                     array_b[:-idx].all())
                    self.assertEqual(array_a[idx:].all(),
                                     array_b[idx:].all())
                    self.assertEqual(array_a[-idx:].all(),
                                     array_b[-idx:].all())
                array_a = self._reread_nparray()

        def test_assignments(self):
            array_a = self._make_nparray(None, shape=(1000, ),
                                         dtype=np.dtype(np.int64))
            array_b = np.array([0] * 1000)
            for idx in range(0, 1000):
                array_a[idx] = idx
                array_b[idx] = idx
            self.assertEqual(array_a[:].all(), array_b[:].all())

        def test_sliced_assignments(self):
            array_a = self._make_nparray(None, shape=(1000, ),
                                         dtype=np.dtype(np.int64))
            array_b = np.array([0] * 1000)
            for idx in range(0, 1000, 100):
                array_a[idx:idx + 100] = 100
                array_b[idx:idx + 100] = 100
                self.assertEqual(array_a[:].all(), array_b[:].all())

        def test_transaction_sorting(self):
            array_a = self._make_nparray(range(1000, 0, -1),
                                         dtype=np.dtype(np.int64))
            array_b = np.array(range(1000, 0, -1))
            with self.pop.transaction():
                array_a.snapshot_range()
                array_a.array.sort()
            array_b.sort()
            self.assertEqual(array_a[:].all(), array_b[:].all())
else:
    class TestPersistentNPArray(TestCase):
        pass


if __name__ == '__main__':
    unittest.main()
