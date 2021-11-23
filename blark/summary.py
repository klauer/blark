from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

from . import transform as tf


def _indented_outline(text: Optional[str]) -> Optional[str]:
    text = text_outline(text)
    if text is None:
        return None
    result = textwrap.indent(text, "    ")
    if "\n" in result:
        return "\n" + result
    return result.lstrip()


def text_outline(item):
    if item is None:
        return None

    if is_dataclass(item):
        result = []
        for fld in fields(item):
            if fld.name in ("meta", ):
                continue
            value = _indented_outline(getattr(item, fld.name, None))
            if value is not None:
                result.append(f"{fld.name}: {value}")
        if not result:
            return None
        return "\n".join(result)

    if isinstance(item, (list, tuple)):
        result = []
        for value in item:
            value = _indented_outline(value)
            if value is not None:
                result.append(f"- {value.lstrip()}")
        if not result:
            return None
        return "\n".join(result)

    if isinstance(item, dict):
        result = []
        for key, value in item.items():
            value = _indented_outline(value)
            if value is not None:
                result.append(f"- {key}: {value}")
        return "\n".join(result)

    return str(item)


@dataclass
class Summary:
    comments: List[str]
    pragmas: List[str]
    meta: Optional[tf.Meta] = field(repr=False)

    def __str__(self) -> str:
        return text_outline(self)

    @staticmethod
    def get_meta_kwargs(meta: Optional[tf.Meta]) -> Dict[str, Any]:
        if meta is None:
            return dict(
                comments=[],
                pragmas=[],
                meta=None,
            )

        comments, pragmas = meta.get_comments_and_pragmas()
        return dict(
            comments=comments,
            pragmas=pragmas,
            meta=meta,
        )


@dataclass
class DeclarationSummary(Summary):
    name: str
    location: Optional[str]
    block: str
    type: str
    value: str

    @classmethod
    def from_declaration(
        cls, item: tf.InitDeclaration, block_type: str = "unknown"
    ) -> Dict[str, DeclarationSummary]:
        result = {}
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
            result[name] = DeclarationSummary(
                name=str(name),
                location=str(location) if location else None,
                block=block_type,
                type=str(spec),
                value=value,
                **Summary.get_meta_kwargs(item.meta),
            )
        return result

    @classmethod
    def from_block(
        cls, block: tf.VariableDeclarationBlock
    ) -> Dict[str, DeclarationSummary]:
        result = {}
        for decl in block.items:
            result.update(cls.from_declaration(decl, block_type=type(decl).__name__))
        return result


@dataclass
class ActionSummary(Summary):
    name: str

    @classmethod
    def from_action(cls, action: tf.Action) -> ActionSummary:
        return ActionSummary(
            name=str(action.name),
            **Summary.get_meta_kwargs(action.meta),
        )


@dataclass
class MethodSummary(Summary):
    return_type: Optional[tf.LocatedVariableSpecInit]
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    @classmethod
    def from_method(cls, method: tf.Method) -> MethodSummary:
        summary = MethodSummary(
            return_type=method.return_type,
            **Summary.get_meta_kwargs(method.meta),
        )
        for decl in method.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl))

        return summary


@dataclass
class FunctionBlockSummary(Summary):
    name: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    actions: List[ActionSummary] = field(default_factory=list)
    methods: List[MethodSummary] = field(default_factory=list)

    @classmethod
    def from_function_block(cls, fb: tf.FunctionBlock) -> FunctionBlockSummary:
        summary = FunctionBlockSummary(
            name=fb.name,
            **Summary.get_meta_kwargs(fb.meta),
        )

        for decl in fb.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl))

        return summary


@dataclass
class CodeSummary:
    function_blocks: Dict[str, FunctionBlockSummary] = field(
        default_factory=dict
    )

    def __str__(self):
        return "\n".join(
            f"{name}:\n{fb}"
            for name, fb in self.function_blocks.items()
        )

    @staticmethod
    def from_source(code: Union[tf.SourceCode, tf.SourceCodeItem]) -> CodeSummary:
        result = CodeSummary()
        if isinstance(code, tf.SourceCode):
            items = code.items
        else:
            items = [code]

        last_function_block = None
        for item in items:
            if isinstance(item, tf.FunctionBlock):
                summary = FunctionBlockSummary.from_function_block(item)
                result.function_blocks[item.name] = summary
                last_function_block = summary
            elif isinstance(item, tf.Method):
                if last_function_block is not None:
                    last_function_block.methods.append(
                        MethodSummary.from_method(item)
                    )
            elif isinstance(item, tf.Action):
                if last_function_block is not None:
                    last_function_block.actions.append(
                        ActionSummary.from_action(item)
                    )
        return result
