import sys

member_value_adapter = None

try:
    import torch
except Exception:
    pass

if "torch" in sys.modules:
    from holowizard.core.parameters.type_conversion.member_value_adapter_torch import (
        MemberValueAdapterTorch,
    )

    member_value_adapter = MemberValueAdapterTorch
else:
    from holowizard.core.parameters.type_conversion.member_value_adapter_numpy import (
        MemberValueAdapterNumpy,
    )

    member_value_adapter = MemberValueAdapterNumpy

