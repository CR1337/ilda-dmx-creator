import ctypes

# 'DMX '
DMX_MAGIC = 0x204D5844


class DmxHeader(ctypes.Structure):
    _fields_ = [
        ('magic', ctypes.c_uint32),
        ('padding', ctypes.c_uint16),
        ('universe', ctypes.c_uint16),
        ('elementCount', ctypes.c_uint32),
        ('duration', ctypes.c_uint32)
    ]


class DmxElement(ctypes.Structure):
    _fields_ = [
        ('time', ctypes.c_uint32),
        ('valueAmount', ctypes.c_uint16)
    ]


class DmxValue(ctypes.Structure):
    _fields_ = [
        ('channel', ctypes.c_uint16),
        ('value', ctypes.c_uint8)
    ]