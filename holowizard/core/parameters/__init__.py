import importlib.util

member_value_adapter = None

if importlib.util.find_spec("torch") is not None:
    from holowizard.core.parameters.type_conversion.member_value_adapter_torch import (
        MemberValueAdapterTorch,
    )

    member_value_adapter = MemberValueAdapterTorch
else:
    from holowizard.core.parameters.type_conversion.member_value_adapter_numpy import (
        MemberValueAdapterNumpy,
    )

    member_value_adapter = MemberValueAdapterNumpy
