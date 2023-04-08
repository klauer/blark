from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Callable, List, Optional, Type, Union

from .typing import Self
from .util import AnyPath, SourceType, find_pou_type_and_identifier


@dataclasses.dataclass(frozen=True)
class BlarkSourceLine:
    filename: Optional[pathlib.Path]
    lineno: int
    code: str

    @classmethod
    def from_code(
        cls: Type[Self],
        code: str,
        first_lineno: int = 1,
        filename: Optional[pathlib.Path] = None,
    ) -> list[Self]:
        return [
            BlarkSourceLine(
                code=line,
                lineno=lineno,
                filename=filename,
            )
            for lineno, line in enumerate(code.splitlines(), start=first_lineno)
        ]


@dataclasses.dataclass
class BlarkSourceItem:
    identifier: str
    lines: list[BlarkSourceLine]
    type: SourceType
    grammar_rule: Optional[str]
    implicit_end: Optional[str]
    user: Optional[Any] = None

    def get_filenames(self) -> set[pathlib.Path]:
        return {line.filename for line in self.lines if line.filename is not None}

    def get_code_and_line_map(
        self,
        include_end: bool = True,
        blark_lineno: int = 1,
    ) -> tuple[str, dict[int, int]]:
        code = "\n".join(line.code for line in self.lines)
        line_map = {
            blark_line: line.lineno
            for blark_line, line in enumerate(self.lines, start=blark_lineno)
        }
        if line_map and self.implicit_end and include_end:
            line_map[max(line_map) + 1] = max(line_map.values())
            code = "\n".join((code, self.implicit_end))
        return code, line_map


@dataclasses.dataclass
class BlarkCompositeSourceItem:
    identifier: str
    filename: Optional[pathlib.Path]  # primary filename?
    parts: list[Union[BlarkSourceItem, BlarkCompositeSourceItem]]
    user: Optional[Any] = None

    @property
    def lines(self) -> list[BlarkSourceLine]:
        lines = []
        for part in self.parts:
            lines.extend(part.lines)
        return lines

    def get_code_and_line_map(
        self,
        include_end: bool = True,
        blark_lineno: int = 1,
    ) -> tuple[str, dict[int, int]]:
        line_to_file_line = {}
        result = []
        for part in self.parts:
            code, inner_line_map = part.get_code_and_line_map(
                include_end=include_end,
                blark_lineno=blark_lineno,
            )
            if code:
                result.append(code)
                line_to_file_line.update(inner_line_map)
                blark_lineno = max(line_to_file_line) + 1

        return "\n".join(result), line_to_file_line

    def get_filenames(self) -> set[pathlib.Path]:
        return {
            line.filename
            for part in self.parts
            for line in part.lines
            if line.filename is not None
        }


handlers = {}

Handler = Callable[
    [pathlib.Path], List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]
]


class UnsupportedFileFormatError(Exception):
    ...


def register_file_handler(extension: str, handler: Handler):
    handlers[extension.lower()] = handler


def load_file_by_name(
    filename: AnyPath,
) -> list[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
    filename = pathlib.Path(filename).expanduser().resolve()
    try:
        handler = handlers[filename.suffix.lower()]
    except KeyError:
        raise UnsupportedFileFormatError(
            f"Unable to find a handler for {filename} based on file extension. "
            f"Supported extensions: {list(handlers)}"
        ) from None

    return handler(filename)


def plain_file_loader(
    filename: pathlib.Path,
) -> List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
    with open(filename, "rt") as fp:
        contents = fp.read()

    source_type, identifier = find_pou_type_and_identifier(contents)
    if source_type is None:
        return []
    item = BlarkSourceItem(
        identifier=identifier,
        lines=[
            BlarkSourceLine(filename=filename, lineno=lineno, code=line)
            for lineno, line in enumerate(contents.splitlines(), 1)
        ],
        type=source_type,
        grammar_rule=source_type.get_grammar_rule(),
        implicit_end=None,  # <-- assume this is already specified
    )
    return [item]


register_file_handler(".txt", plain_file_loader)
register_file_handler(".st", plain_file_loader)