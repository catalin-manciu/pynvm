import numpy as np
import sys
from _pmem import ffi

NP_POBJPTR_ARRAY_TYPE_NUM = 70
NP_POBJPTR_STR_TYPE_NUM = 71


class PersistentNPArray(object):
    def _validate_keywords(self, **kw):
        mandatory_fields = ["dtype"]
        for field in mandatory_fields:
            if field not in kw:
                raise TypeError("Missing '%s' parameter" % (field))
        if type(kw["dtype"]) is not np.dtype:
            raise TypeError("'dtype' argument must be of type np.dtype")
        if not kw["dtype"].isbuiltin:
            raise TypeError("'dtype' argument must represent a builtin type")
        if "shape" in kw:
            shape = kw["shape"]
            if type(shape) is int:
                if shape <= 0:
                    raise ValueError(
                        "'shape' should be a strictly positive value")
                kw["shape"] = (shape, )
            elif type(shape) is tuple:
                if len(shape) != 1 or\
                 type(shape[0]) is not int or shape[0] <= 0:
                    raise ValueError(
                        "'shape' should contain a single,"
                        " strictly positive value")
            else:
                raise TypeError("'shape' is an unsupported"
                                "data type: %s" % type(shape))

    def __init__(self, *args, **kw):
        self._validate_keywords(**kw)

        data_size = 0
        data_init = None

        if len(args) and hasattr(args[0], "__len__") and len(args[0]):
            data_size = len(args[0])
            data_init = args[0]

        if "shape" in kw:
            data_size = max(data_size, kw["shape"][0])
            data_shape = kw["shape"]
        else:
            data_shape = (data_size,)

        data_type = kw["dtype"]
        data_size *= data_type.itemsize
        data_type_name = str(data_type)

        if sys.version_info[0] > 2:
            data_type_name = data_type_name.encode('utf-8')

        mm = self._p_mm
        with mm.transaction():
            self._body.data = mm.zalloc(data_size,
                                        type_num=NP_POBJPTR_ARRAY_TYPE_NUM)
            data_buffer = ffi.buffer(mm.direct(self._body.data), data_size)
            self._body.dtypestr = mm.zalloc(
                        len(data_type_name) + 1,
                        type_num=NP_POBJPTR_STR_TYPE_NUM)
            str_buff = ffi.cast('char *', mm.direct(self._body.dtypestr))
            ffi.buffer(str_buff, len(data_type_name))[:] = data_type_name
            self.array = np.ndarray(shape=data_shape,
                                    dtype=data_type,
                                    buffer=data_buffer)
            ob = ffi.cast('PVarObject *', mm.direct(self._p_oid))
            ob.ob_size = data_shape[0]

            if data_init is not None:
                self.array[:data_size] = data_init

    def _p_new(self, manager):
        mm = self._p_mm = manager
        with mm.transaction():
            self._p_oid = mm.zalloc(ffi.sizeof('PNPArrayObject'))
            ob = ffi.cast('PObject *', mm.direct(self._p_oid))
            ob.ob_type = mm._get_type_code(self.__class__)
            ob = ffi.cast('PVarObject *', mm.direct(self._p_oid))
            ob.ob_size = 0
            self._body = ffi.cast('PNPArrayObject *', mm.direct(self._p_oid))
            self._body.dtypestr = mm.OID_NULL
            self._body.data = mm.OID_NULL
            self.array = None

    def _p_resurrect(self, manager, oid):
        mm = self._p_mm = manager
        self._p_oid = oid
        ob = ffi.cast('PVarObject *', mm.direct(self._p_oid))
        self._body = ffi.cast('PNPArrayObject *', mm.direct(oid))
        if mm.otuple(self._body.dtypestr) != mm.OID_NULL and\
           mm.otuple(self._body.data) != mm.OID_NULL:
            data_type_name = ffi.string(ffi.cast('char *',
                                        mm.direct(self._body.dtypestr)))
            data_type = np.dtype(data_type_name)
            data_buffer = ffi.buffer(mm.direct(self._body.data),
                                     data_type.itemsize * ob.ob_size)
            data_shape = (ob.ob_size, )
            self.array = np.ndarray(shape=data_shape,
                                    dtype=data_type,
                                    buffer=data_buffer)

    def _p_substructures(self):
        return ((self._body.data, NP_POBJPTR_ARRAY_TYPE_NUM),
                (self._body.dtypestr, NP_POBJPTR_STR_TYPE_NUM))

    def _p_traverse(self):
        return []

    def _p_deallocate(self):
        mm = self._p_mm
        if mm.otuple(self._body.dtypestr) != mm.OID_NULL:
            mm.free(self._body.dtypestr)
        if mm.otuple(self._body.data) != mm.OID_NULL:
            mm.free(self._body.data)

    def _normalize_index(self, index):
        nparr_len = self.array.shape[0]
        if isinstance(index, slice):
            start = index.start
            if start is None:
                start = 0
            elif start < 0:
                new_start = start + nparr_len
                if new_start < 0:
                    raise IndexError(
                        "index %d is out of bounds for axis 0 with size %d" %
                        (start, nparr_len))
                start = new_start
            else:
                if start >= nparr_len:
                    raise IndexError(
                        "index %d is out of bounds for axis 0 with size %d" %
                        (start, nparr_len))
            stop = index.stop
            if stop is None:
                stop = nparr_len
            elif stop < 0:
                new_stop = stop + nparr_len
                if new_stop < 0:
                    raise IndexError(
                        "index %d is out of bounds for axis 0 with size %d" %
                        (stop, nparr_len))
                stop = new_stop
            else:
                if stop > nparr_len:
                    raise IndexError(
                        "index %d is out of bounds for axis 0 with size %d" %
                        (stop, nparr_len))
            if index.step is None or index.step > 0:
                return (start, stop)
            return (stop, start)
        else:
            try:
                index = int(index)
            except ValueError:
                raise NotImplementedError("Not a recognized index type")
            if index < 0:
                index = nparr_len + index
            if index < 0 or index >= nparr_len:
                raise IndexError(
                        "index %d is out of bounds for axis 0 with size %d" %
                        (index, nparr_len))
            return (index, index + 1)

    def _snapshot_slice(self, index):
        mm = self._p_mm
        snapshot_bounds = self._normalize_index(index)
        bounds_size = snapshot_bounds[1] - snapshot_bounds[0]
        if bounds_size > 0:
            data_item_size = self.array.dtype.itemsize
            mm.snapshot_range(mm.direct(self._body.data),
                              data_item_size * bounds_size)

    def snapshot_range(self, start_index=0, end_index=None):
        _slice = slice(start_index, end_index)
        self._snapshot_slice(_slice)

    def snapshot_index(self, index):
        self._snapshot_slice(index)

    def __setitem__(self, index, value):
        with self._p_mm.transaction():
            self._snapshot_slice(index)
            self.array.__setitem__(index, value)

    def __getitem__(self, index):
        return self.array.__getitem__(index)

    def __len__(self):
        return self._size
