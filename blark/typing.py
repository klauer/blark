from __future__ import annotations

import typing
from typing import Union, overload

__all__ = ["Self"]


if typing.TYPE_CHECKING:
    from .util import BlarkCompositeSourceItem, BlarkSourceItem


try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@runtime_checkable
class ContainsBlarkCode(Protocol):
    @overload
    def to_blark(self) -> list[BlarkSourceItem]:
        ...

    @overload
    def to_blark(self) -> list[BlarkCompositeSourceItem]:
        ...

    @overload
    def to_blark(self) -> list[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
        ...
