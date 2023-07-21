from __future__ import annotations

import pathlib
import typing
from typing import Callable, Optional, Union, overload

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

__all__ = ["Self", "Literal"]


if typing.TYPE_CHECKING:
    from .input import BlarkCompositeSourceItem, BlarkSourceItem


try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


#: Support both pathlib paths and regular strings with AnyPath:
AnyPath = Union[str, pathlib.Path]
AnyBlarkSourceItem = Union["BlarkCompositeSourceItem", "BlarkSourceItem"]
Preprocessor = Callable[[str], str]


DeclarationOrImplementation = Literal["declaration", "implementation"]


@runtime_checkable
class ContainsBlarkCode(Protocol):
    """Indicates that the given class can emit blark-compatible source items."""

    @overload
    def to_blark(self) -> list[BlarkSourceItem]:
        ...

    @overload
    def to_blark(self) -> list[BlarkCompositeSourceItem]:
        ...

    @overload
    def to_blark(self) -> list[AnyBlarkSourceItem]:
        ...


# TODO: these got refactored out; any use for them?


@runtime_checkable
class SupportsRewrite(Protocol):
    def rewrite_code(self, identifier: Optional[str], contents: str):
        ...


@runtime_checkable
class SupportsWrite(Protocol):
    @overload
    def to_file_contents(self, **kwargs) -> str:
        ...

    @overload
    def to_file_contents(self, **kwargs) -> bytes:
        ...


@runtime_checkable
class SupportsSaveToPath(Protocol):
    def save_to(self, path: AnyPath, **kwargs) -> None:
        ...
