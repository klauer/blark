from __future__ import annotations

import typing
from typing import Union, overload

if typing.TYPE_CHECKING:
    from .util import BlarkCompositeSourceItem, BlarkSourceItem


try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class ContainsBlarkCode(Protocol):
    @overload
    def to_blark(self) -> list[BlarkCompositeSourceItem]:
        ...

    @overload
    def to_blark(self) -> list[BlarkSourceItem]:
        ...

    @overload
    def to_blark(self) -> list[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
        ...


BlarkLineToFileLine = dict[int, int]
