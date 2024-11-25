import ctypes


ILDX_MAGIC = 0x494C4458
ILDA_MAGIC = 0x494C4441

ILDX_STATUS_CODE_LAST_POINT_MASK = 0b10000000;
ILDX_STATUS_CODE_BLANKING_MASK = 0b01000000;


class IldxHeader(ctypes.BigEndianStructure):
    _fields_ = [
        ('ildxMagic', ctypes.c_uint32),
        ('starttime', ctypes.c_uint8 * 3),
        ('formatCode', ctypes.c_uint8),
        ('frameName', ctypes.c_char * 8),
        ('companyName', ctypes.c_char * 8),
        ('numberOfRecords', ctypes.c_uint16),
        ('frameOrPaletteNumber', ctypes.c_uint16),
        ('totalFrames', ctypes.c_uint16),
        ('projectorNumber', ctypes.c_uint8),
        ('framesPerSecondOrFrameAmount', ctypes.c_uint8)
    ]


class Ilda3dIndexedRecord(ctypes.BigEndianStructure):
    _fields_ = [
        ('x', ctypes.c_int16),
        ('y', ctypes.c_int16),
        ('z', ctypes.c_int16),
        ('statusCode', ctypes.c_uint8),
        ('colorIndex', ctypes.c_uint8)
    ]


class Ilda2dIndexedRecord(ctypes.BigEndianStructure):
    _fields_ = [
        ('x', ctypes.c_int16),
        ('y', ctypes.c_int16),
        ('statusCode', ctypes.c_uint8),
        ('colorIndex', ctypes.c_uint8)
    ]


class IldaColorPalette(ctypes.BigEndianStructure):
    _fields_ = [
        ('r', ctypes.c_uint8),
        ('g', ctypes.c_uint8),
        ('b', ctypes.c_uint8)
    ]


class Ilda3dTrueColorRecord(ctypes.BigEndianStructure):
    _fields_ = [
        ('x', ctypes.c_int16),
        ('y', ctypes.c_int16),
        ('z', ctypes.c_int16),
        ('statusCode', ctypes.c_uint8),
        ('r', ctypes.c_uint8),
        ('g', ctypes.c_uint8),
        ('b', ctypes.c_uint8)
    ]


class Ilda2dTrueColorRecord(ctypes.BigEndianStructure):
    _fields_ = [
        ('x', ctypes.c_int16),
        ('y', ctypes.c_int16),
        ('statusCode', ctypes.c_uint8),
        ('r', ctypes.c_uint8),
        ('g', ctypes.c_uint8),
        ('b', ctypes.c_uint8)
    ]


def adjust_start_time(time: int) -> bytes:
    bytes_data = time.to_bytes(3, byteorder='big')
    bytes_array = (ctypes.c_uint8 * 3)(*bytes_data)
    return bytes_array


def zero_start_time() -> bytes:
    return (ctypes.c_uint8 * 3)(*bytes([0, 0, 0]))
