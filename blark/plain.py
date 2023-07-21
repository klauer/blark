"""Plain (plain text or human-readble flat structured text) file loader."""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, List, Optional, Union

from .input import (BlarkCompositeSourceItem, BlarkSourceItem,
                    register_input_handler)
from .output import OutputBlock, register_output_handler
from .util import AnyPath, SourceType, find_pou_type_and_identifier


@dataclasses.dataclass
class PlainFileLoader:
    filename: pathlib.Path
    raw_source: str
    source_type: SourceType = SourceType.general
    identifier: Optional[str] = None
    formatted_code: Optional[str] = None

    def rewrite_code(self, identifier: str, code: str) -> None:
        self.formatted_code = code

    def save_to(self, path: AnyPath) -> None:
        code = self.to_file_contents()
        with open(path, "wt") as fp:
            print(code, file=fp)

    def to_file_contents(self) -> str:
        if self.formatted_code is not None:
            return self.formatted_code
        return self.raw_source

    @classmethod
    def load(
        cls,
        filename: pathlib.Path,
    ) -> List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
        with open(filename, "rt") as fp:
            contents = fp.read()

        source_type, identifier = find_pou_type_and_identifier(contents)
        # if source_type is None:
        #     return []
        source_type = SourceType.general
        # We'll pick the first identifier here only. IEC source (as what blark
        # accepts for iput) it's allowed to have multiple declarations in the same
        # file.
        # TODO: If people want to use it like this, we could pre-parse the file for
        # all identifiers and return a BlarkCompositeSourceItem.
        # As-is, the focus is now on loading TwinCAT XML files directly.
        loader = cls(
            filename=filename,
            identifier=identifier,
            source_type=source_type,
            raw_source=contents,
        )
        return [
            BlarkSourceItem.from_code(
                code=contents,
                source_type=source_type,
                identifier=identifier or "",
                first_lineno=1,
                user=loader,
                filename=filename,
            )
        ]

    @staticmethod
    def save(
        user: Any,
        source_filename: Optional[pathlib.Path],
        parts: List[OutputBlock],
    ) -> str:
        return "\n\n".join(part.code for part in parts)


def _register():
    """Register the plain file handlers."""
    register_input_handler("plain", PlainFileLoader.load)
    register_input_handler(".txt", PlainFileLoader.load)
    register_input_handler(".st", PlainFileLoader.load)

    register_output_handler("plain", PlainFileLoader.save)
    register_output_handler(".st", PlainFileLoader.save)
    register_output_handler(".txt", PlainFileLoader.save)
