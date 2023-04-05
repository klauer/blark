from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional, Type, Union

from .typing import Self
from .util import SourceType


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
        if self.implicit_end and include_end:
            line_map[max(line_map) + 1] = max(line_map.values())
            code = "\n".join((code, self.implicit_end))
        return code, line_map


@dataclasses.dataclass
class BlarkCompositeSourceItem:
    identifier: str
    filename: pathlib.Path  # primary filename?
    parts: list[Union[BlarkSourceItem, BlarkCompositeSourceItem]]

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
