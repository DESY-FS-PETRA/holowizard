import sys
import numpy as np

from holowizard.core.parameters.type_conversion.member_value_adapter_interface import (
    MemberValueAdapterInterface,
)


class MemberValueAdapterNumpy(MemberValueAdapterInterface):
    @staticmethod
    def get_array(data):
        return MemberValueAdapterNumpy.get_array_complex(data)

    @staticmethod
    def get_array_complex(data):
        if data is None:
            return None
        elif type(data) is np.ndarray:
            return data.astype(np.complex64)
        else:
            raise TypeError("Expected numpy array but got ", type(data)).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_array_float(data):
        if data is None:
            return None
        elif type(data) is np.ndarray:
            return data.astype(np.single)
        else:
            raise TypeError("Expected numpy array but got ", type(data)).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_array_ushort(data):
        if data is None:
            return None
        elif type(data) is np.ndarray:
            return data.astype(np.uint16)
        else:
            raise TypeError("Expected numpy array but got ", type(data)).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_numpy_array(data):
        return MemberValueAdapterNumpy.get_array(data)

    @staticmethod
    def get_numpy_array_float(data):
        return MemberValueAdapterNumpy.get_array_float(data)

    @staticmethod
    def get_numpy_array_ushort(data):
        return MemberValueAdapterNumpy.get_array_ushort(data)

    @staticmethod
    def get_numpy_array_complex(data):
        return MemberValueAdapterNumpy.get_array_complex(data)

    @staticmethod
    def get_float(data):
        if type(data) is float or type(data) is np.float64:
            return data
        elif type(data) is int:
            return float(data)
        else:
            raise TypeError("Expected float or int but got ", type(data)).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_complex(data):
        if data is None:
            return None

        if np.iscomplexobj(data):
            return data
        else:
            raise TypeError(
                "Expected complex for but got ",
                type(data),
            ).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_tuple(data):
        if type(data) is tuple:
            return data
        if type(data) is list:
            return tuple(data)
        else:
            raise TypeError("Expected tuple or list but got", type(data)).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_array_wavefield_from_refractive(data):
        if data is None:
            return None
        elif type(data) is np.ndarray:
            return np.exp(1j * data.astype(np.single))
        else:
            raise TypeError("Expected numpy array but got ", type(data)).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_array_refractive_from_wavefield(data):
        return np.angle(data) - 1j * np.log(np.abs(data))

    @staticmethod
    def get_range_tuple_list(data):
        if (
            type(data) is list
            and len(data) == 2
            and type(data[0]) is tuple
            and type(data[1]) is tuple
            and len(data[0]) == 2
            and len(data[1]) == 2
        ):
            return data
        elif (
            type(data) is tuple
            and len(data) == 2
            and type(data[0]) is tuple
            and type(data[1]) is tuple
            and len(data[0]) == 2
            and len(data[1]) == 2
        ):
            return list(data)
        elif (
            type(data) is list
            and len(data) == 2
            and type(data[0]) is list
            and type(data[1]) is list
            and len(data[0]) == 2
            and len(data[1]) == 2
        ):
            return list([tuple(data[0]), tuple(data[1])])
        else:
            raise TypeError(
                "Expected tuple list [(int,int),(int,int)] or tuple tuple  ((int,int),(int)(int))"
            ).with_traceback(sys.exc_info()[2])

    @staticmethod
    def get_string(data):
        if type(data) is str:
            return data
        else:
            raise TypeError("Expected string but got ", type(data)).with_traceback(sys.exc_info()[2])
