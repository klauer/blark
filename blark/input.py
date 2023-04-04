from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional, Union

from . import util
from .typing import BlarkLineToFileLine
from .util import SourceType


@dataclasses.dataclass
class BlarkSourceItem:
    file: Optional[pathlib.Path]
    line: int
    type: SourceType
    code: str
    implicit_end: Optional[str]
    grammar_rule: Optional[str]

    def get_files(self) -> set[pathlib.Path]:
        if self.file is not None:
            return {self.file}
        return set()

    def get_code(
        self,
        include_end: bool = False,
        blark_lineno: int = 1,
    ) -> tuple[str, BlarkLineToFileLine]:
        if include_end and self.implicit_end:
            code = "\n".join((self.code, self.implicit_end))
        else:
            code = self.code

        code = code.rstrip()
        line_map = util._build_source_to_file_line_map(
            file_lineno=self.line,
            blark_lineno=blark_lineno,
            source=code,
        )
        return code, line_map


@dataclasses.dataclass
class BlarkCompositeSourceItem:
    type: SourceType
    parts: list[Union[BlarkSourceItem, BlarkCompositeSourceItem]]
    implicit_end: Optional[str]
    grammar_rule: Optional[str]

    def get_files(self) -> set[pathlib.Path]:
        files = set()
        for part in self.parts:
            files |= part.get_files()
        return files

    def get_code(
        self,
        include_end: bool = False,
        blark_lineno: int = 1,
    ) -> tuple[str, BlarkLineToFileLine]:
        line_to_file_line = {}
        result = []
        for part in self.parts:
            code, inner_line_map = part.get_code(
                include_end=False,
                blark_lineno=blark_lineno,
            )
            if code:
                result.append(code)
                line_to_file_line.update(inner_line_map)
                blark_lineno = max(line_to_file_line) + 1
            # if part.implicit_end:
            #     result.append(part.implicit_end)
        if self.implicit_end and include_end:
            result.append(self.implicit_end)
            # This doesn't exist in the file...
            line_to_file_line[blark_lineno] = max(line_to_file_line.values())

        return "\n".join(result), line_to_file_line
