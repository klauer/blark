from __future__ import annotations

import dataclasses
import pathlib
import typing
from typing import Any, Callable, Dict, List, Optional, Union

if typing.TYPE_CHECKING:
    from .parse import ParseResult


@dataclasses.dataclass
class OutputBlock:
    #: The (optionally modified) code to write as output.
    code: str
    #: Metadata to add/modify.
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: The origin of the above code block.
    origin: Optional[ParseResult] = None


OutputHandler = Callable[
    [
        # The "user" object that parsed the input file.
        # For now, we're assuming that all output is based on existing
        # files that were parsed.
        object,
        # The input filename, if it exists.
        Optional[pathlib.Path],
        # The code the user wants to write to the file.
        List[OutputBlock]
    ],
    # The file contents to be written.
    Union[str, bytes],
]

handlers: Dict[str, OutputHandler] = {}


def register_output_handler(name: str, handler: OutputHandler):
    handlers[name.lower()] = handler


def get_handler_by_name(name: str) -> OutputHandler:
    try:
        return handlers[name.lower()]
    except KeyError:
        available = ", ".join(sorted(handlers))
        raise ValueError(
            f"Unknown output handler {name!r}. "
            f"Available handlers include: {available}"
        ) from None
