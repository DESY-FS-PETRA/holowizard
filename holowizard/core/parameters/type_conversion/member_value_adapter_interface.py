from abc import ABC, abstractmethod


class MemberValueAdapterInterface(ABC):
    @staticmethod
    @abstractmethod
    def get_array(data):
        ...

    @staticmethod
    @abstractmethod
    def get_array_complex(data):
        ...

    @staticmethod
    @abstractmethod
    def get_array_float(data):
        ...

    @staticmethod
    @abstractmethod
    def get_array_ushort(data):
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_array(data):
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_array_complex(data):
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_array_float(data):
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_array_ushort(data):
        ...

    @staticmethod
    @abstractmethod
    def get_float(data):
        ...

    @staticmethod
    @abstractmethod
    def get_complex(data):
        ...

    @staticmethod
    @abstractmethod
    def get_string(data):
        ...

    @staticmethod
    @abstractmethod
    def get_tuple(data):
        ...

    @staticmethod
    @abstractmethod
    def get_array_wavefield_from_refractive(data):
        ...

    @staticmethod
    @abstractmethod
    def get_array_refractive_from_wavefield(data):
        ...

    @staticmethod
    @abstractmethod
    def get_range_tuple_list(data):
        ...
