from abc import ABC, abstractmethod
from typing import Any, TypeVar

HPCS = TypeVar("HPCS", bound="HpcSerializable")

class HpcSerializable(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        ...

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        ...

    @classmethod
    @abstractmethod
    def from_json(cls: type[HPCS], json_data: str, **kwargs: Any) -> HPCS: ...