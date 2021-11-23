from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

from . import transform as tf


def _get_comments(item):
    try:
        return item.meta.comments
    except AttributeError:
        return []


@dataclass
class Declaration:
    name: str
    location: str
    block: str
    type: str
    value: str
    comments: List[str]
    pragmas: List[str]


@dataclass
class FunctionBlockSummary:
    name: str
    comments: List[str] = field(default_factory=list)
    declarations: Dict[str, Declaration] = field(default_factory=dict)
    # actions:
    # methods:

    @classmethod
    def from_function_block(cls, fb: tf.FunctionBlock) -> FunctionBlockSummary:
        summary = FunctionBlockSummary(name=fb.name, comments=_get_comments(fb))

        for decl in fb.declarations:
            for item in decl.items:
                # OK, a bit lazy for now
                try:
                    spec = item.init.spec
                except AttributeError:
                    spec = "?"

                try:
                    value = item.init.value
                except AttributeError:
                    value = ""

                for var in item.variables:
                    name = getattr(var, "name", var)
                    location = getattr(var, "location", None)
                    comments, pragmas = item.meta.get_comments_and_pragmas()
                    summary.declarations[name] = Declaration(
                        name=name,
                        location=location,
                        block=type(decl).__name__,
                        type=str(spec),
                        value=value,
                        comments=comments,
                        pragmas=pragmas,
                    )

        return summary


@dataclass
class CodeSummary:
    function_blocks: Dict[str, FunctionBlockSummary] = field(
        default_factory=dict
    )

    @staticmethod
    def from_source(code: Union[tf.SourceCode, tf.SourceCodeItem]) -> CodeSummary:
        result = CodeSummary()
        if isinstance(code, tf.SourceCode):
            items = code.items
        else:
            items = [code]

        for item in items:
            if isinstance(item, tf.FunctionBlock):
                result.function_blocks[item.name] = FunctionBlockSummary.from_function_block(item)
        return result
