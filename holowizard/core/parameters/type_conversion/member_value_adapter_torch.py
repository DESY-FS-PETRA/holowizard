import sys
import numpy as np

import holowizard.interface
from holowizard.core.parameters.type_conversion.member_value_adapter_numpy import (
    MemberValueAdapterNumpy,
)

try:
    import torch
except Exception:
    pass

if "torch" in sys.modules:

    class MemberValueAdapterTorch(MemberValueAdapterNumpy):
        @staticmethod
        def get_array(data):
            return MemberValueAdapterTorch.get_array_complex(data)

        @staticmethod
        def get_array_complex(data):
            if data is None:
                return None
            elif type(data) is torch.Tensor:
                return data.to(torch.complex64)
            elif type(data) is np.ndarray:
                return torch.from_numpy(data.astype(np.complex64)).to(
                    holowizard.interface.torch_running_device
                )
            else:
                raise TypeError(
                    "Expected torch tensor or numpy array but got ", type(data)
                ).with_traceback(sys.exc_info()[2])

        @staticmethod
        def get_array_float(data):
            if data is None:
                return None
            elif type(data) is torch.Tensor:
                return data.to(torch.float)
            elif type(data) is np.ndarray:
                return torch.from_numpy(data.astype(np.single)).to(
                    holowizard.interface.torch_running_device
                )
            else:
                raise TypeError(
                    "Expected torch tensor or numpy array but got ", type(data)
                ).with_traceback(sys.exc_info()[2])

        @staticmethod
        def get_array_ushort(data):
            if data is None:
                return None
            elif type(data) is torch.Tensor:
                return data.to(torch.uint16)
            elif type(data) is np.ndarray:
                return torch.from_numpy(data.astype(np.uint16)).to(
                    holowizard.interface.torch_running_device
                )
            else:
                raise TypeError(
                    "Expected torch tensor or numpy array but got ", type(data)
                ).with_traceback(sys.exc_info()[2])

        @staticmethod
        def get_numpy_array(data):
            return MemberValueAdapterTorch.get_numpy_array_complex(data)

        @staticmethod
        def get_numpy_array_complex(data):
            if data is None:
                return None
            elif type(data) is torch.Tensor:
                return np.ascontiguousarray(data.cpu().numpy().astype(np.complex64))
            elif type(data) is np.ndarray:
                return data.astype(np.complex64)
            else:
                raise TypeError(
                    "Expected torch tensor or numpy array but got ", type(data)
                ).with_traceback(sys.exc_info()[2])

        @staticmethod
        def get_numpy_array_float(data):
            if data is None:
                return None
            elif type(data) is torch.Tensor:
                return np.ascontiguousarray(data.cpu().numpy().astype(np.single))
            elif type(data) is np.ndarray:
                return data.astype(np.single)
            else:
                raise TypeError(
                    "Expected torch tensor or numpy array but got ", type(data)
                ).with_traceback(sys.exc_info()[2])

        @staticmethod
        def get_numpy_array_ushort(data):
            if data is None:
                return None
            elif type(data) is torch.Tensor:
                return np.ascontiguousarray(data.cpu().numpy().astype(np.uint16))
            elif type(data) is np.ndarray:
                return data.astype(np.uint16)
            else:
                raise TypeError(
                    "Expected torch tensor or numpy array but got ", type(data)
                ).with_traceback(sys.exc_info()[2])

        @staticmethod
        def get_array_wavefield_from_refractive(data):
            if data is None:
                return None

            return torch.exp(1j * MemberValueAdapterTorch.get_array(data))

        @staticmethod
        def get_array_refractive_from_wavefield(data):
            if data is None:
                return None

            data_torch = MemberValueAdapterTorch.get_array(data)
            return torch.angle(data_torch) - 1j * torch.log(torch.abs(data_torch))
