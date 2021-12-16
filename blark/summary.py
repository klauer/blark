from __future__ import annotations

import textwrap
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

from . import transform as tf


def _indented_outline(item: Any, indent: str = "    ") -> Optional[str]:
    """Outline and indent the given item."""
    text = text_outline(item)
    if text is None:
        return None
    result = textwrap.indent(text, indent)
    if "\n" in result:
        return "\n" + result
    return result.lstrip()


def text_outline(item: Any) -> Optional[str]:
    """
    Get a generic multiline string representation of the given object.

    Attempts to include field information for dataclasses, put list items
    on separate lines, and generally keep sensible indentation.

    Parameters
    ----------
    item : Any
        The item to outline.

    Returns
    -------
    formatted : str or None
        The formatted result.
    """
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
    """Base class for summary objects."""
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


if hasattr(typing, "Literal"):
    from typing import Literal
    LocationType = Union[Literal["input"], Literal["output"], Literal["memory"]]
else:
    LocationType = str


@dataclass
class DeclarationSummary(Summary):
    """Summary representation of a single declaration."""
    name: str
    parent: str
    location: Optional[str]
    block: str
    base_type: str
    type: str
    value: Optional[str]

    @property
    def qualified_name(self) -> str:
        """Qualified name including parent. For example, ``fbName.DeclName``."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def location_type(self) -> Optional[LocationType]:
        """If located, one of {'input', 'output', 'memory"}."""
        if not self.location:
            return None

        location = self.location.upper()
        if location.startswith("%I"):
            return "input"
        if location.startswith("%Q"):
            return "output"
        if location.startswith("%M"):
            return "memory"
        return None

    @classmethod
    def from_declaration(
        cls,
        item: tf.InitDeclaration,
        parent: Optional[
            Union[tf.Function, tf.Method, tf.FunctionBlock, tf.StructureTypeDeclaration]
        ] = None,
        block_header: str = "unknown",
    ) -> Dict[str, DeclarationSummary]:
        result = {}
        # OK, a bit lazy for now
        try:
            spec = item.init.spec
        except AttributeError:
            spec = item.init
            spec = getattr(spec, "name", spec)

        if isinstance(spec, (str, tf.SimpleVariable)):  # TODO
            base_type = str(spec)
        elif hasattr(spec, "type_name"):
            base_type = str(spec.type_name)
        elif hasattr(spec, "type"):
            if hasattr(spec.type, "type_name"):
                base_type = spec.type.type_name
            else:
                base_type = str(spec.type)
        else:
            raise ValueError(f"TODO: {type(spec)}")

        try:
            value = item.init.value
        except AttributeError:
            value = item.init

        for var in item.variables:
            name = getattr(var, "name", var)
            location = getattr(var, "location", None)
            result[name] = DeclarationSummary(
                name=str(name),
                location=str(location).replace("AT ", "") if location else None,
                block=block_header,
                type=str(spec),
                base_type=base_type,
                value=value,
                parent=parent.name if parent is not None else "",
                **Summary.get_meta_kwargs(item.meta),
            )
        return result

    @classmethod
    def from_block(
        cls,
        block: tf.VariableDeclarationBlock,
        parent: Union[tf.Function, tf.Method, tf.FunctionBlock],
    ) -> Dict[str, DeclarationSummary]:
        result = {}
        for decl in block.items:
            result.update(
                cls.from_declaration(decl, parent=parent, block_header=block.block_header)
            )
        return result


@dataclass
class ActionSummary(Summary):
    """Summary representation of a single action."""
    name: str
    source_code: str

    @classmethod
    def from_action(cls, action: tf.Action, source_code: Optional[str] = None) -> ActionSummary:
        if source_code is None:
            source_code = str(action)

        return ActionSummary(
            name=str(action.name),
            source_code=source_code,
            **Summary.get_meta_kwargs(action.meta),
        )


@dataclass
class MethodSummary(Summary):
    """Summary representation of a single method."""
    name: str
    return_type: Optional[str]
    source_code: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    @property
    def declarations_by_block(self) -> Dict[str, Dict[str, DeclarationSummary]]:
        result = {}
        for decl in self.declarations.values():
            result.setdefault(decl.block, {})[decl.name] = decl
        return result

    @classmethod
    def from_method(cls, method: tf.Method, source_code: Optional[str] = None) -> MethodSummary:
        if source_code is None:
            source_code = str(method)

        summary = MethodSummary(
            name=method.name,
            return_type=str(method.return_type) if method.return_type else None,
            source_code=source_code,
            **Summary.get_meta_kwargs(method.meta),
        )
        for decl in method.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl, parent=method))

        return summary


@dataclass
class FunctionSummary(Summary):
    """Summary representation of a single function."""
    name: str
    return_type: Optional[str]
    source_code: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    @property
    def declarations_by_block(self) -> Dict[str, Dict[str, DeclarationSummary]]:
        result = {}
        for decl in self.declarations.values():
            result.setdefault(decl.block, {})[decl.name] = decl
        return result

    @classmethod
    def from_function(
        cls, func: tf.Function, source_code: Optional[str] = None
    ) -> FunctionSummary:
        if source_code is None:
            source_code = str(func)

        summary = FunctionSummary(
            name=func.name,
            return_type=str(func.return_type) if func.return_type else None,
            source_code=source_code,
            **Summary.get_meta_kwargs(func.meta),
        )

        for decl in func.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl, parent=func))

        return summary


@dataclass
class FunctionBlockSummary(Summary):
    """Summary representation of a single function block."""
    name: str
    source_code: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    actions: List[ActionSummary] = field(default_factory=list)
    methods: List[MethodSummary] = field(default_factory=list)

    @property
    def declarations_by_block(self) -> Dict[str, Dict[str, DeclarationSummary]]:
        result = {}
        for decl in self.declarations.values():
            result.setdefault(decl.block, {})[decl.name] = decl
        return result

    @classmethod
    def from_function_block(
        cls, fb: tf.FunctionBlock, source_code: Optional[str] = None
    ) -> FunctionBlockSummary:
        if source_code is None:
            source_code = str(fb)

        summary = FunctionBlockSummary(
            name=fb.name,
            source_code=source_code,
            **Summary.get_meta_kwargs(fb.meta),
        )

        for decl in fb.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl, parent=fb))

        return summary


@dataclass
class DataTypeSummary(Summary):
    """Summary representation of a single function block."""
    name: str
    source_code: str
    type: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    @property
    def declarations_by_block(self) -> Dict[str, Dict[str, DeclarationSummary]]:
        return {
            "STRUCT": self.declarations
        }

    @classmethod
    def from_data_type(
        cls, dtype: tf.TypeDeclarationItem, source_code: Optional[str] = None
    ) -> DataTypeSummary:
        if source_code is None:
            source_code = str(dtype)

        summary = DataTypeSummary(
            name=dtype.name,
            source_code=source_code,
            type=type(dtype).__name__,
            **Summary.get_meta_kwargs(dtype.meta),
        )

        if isinstance(dtype, tf.StructureTypeDeclaration):
            for decl in dtype.declarations:
                summary.declarations.update(
                    DeclarationSummary.from_declaration(decl, parent=dtype, block_header="STRUCT")
                )

        return summary


@dataclass
class CodeSummary:
    """Summary representation of a set of code - functions, function blocks, etc."""
    functions: Dict[str, FunctionSummary] = field(default_factory=dict)
    function_blocks: Dict[str, FunctionBlockSummary] = field(
        default_factory=dict
    )
    data_types: Dict[str, DataTypeSummary] = field(default_factory=dict)

    def __str__(self):
        return "\n".join(
            f"{name}:\n{fb}"
            for name, fb in self.function_blocks.items()
        )

    @staticmethod
    def from_source(code: tf.SourceCode) -> CodeSummary:
        result = CodeSummary()
        code_by_lines = [""] + code.raw_source.splitlines()
        items = code.items

        def get_code_by_meta(meta: Optional[tf.Meta]) -> str:
            if not meta:
                return ""
            return "\n".join(code_by_lines[meta.line:meta.end_line + 1])

        last_function_block = None
        for item in items:
            if isinstance(item, tf.FunctionBlock):
                summary = FunctionBlockSummary.from_function_block(
                    item,
                    source_code=get_code_by_meta(item.meta)
                )
                result.function_blocks[item.name] = summary
                last_function_block = summary
            elif isinstance(item, tf.Function):
                summary = FunctionSummary.from_function(
                    item,
                    source_code=get_code_by_meta(item.meta)
                )
                result.functions[item.name] = summary
                last_function_block = None
            elif isinstance(item, tf.DataTypeDeclaration):
                for subitem in item.items:
                    summary = DataTypeSummary.from_data_type(
                        subitem,
                        source_code=get_code_by_meta(subitem.meta)
                    )
                    result.data_types[subitem.name] = summary
                last_function_block = None
            elif isinstance(item, tf.Method):
                if last_function_block is not None:
                    last_function_block.methods.append(
                        MethodSummary.from_method(
                            item,
                            source_code=get_code_by_meta(item.meta)
                        )
                    )
            elif isinstance(item, tf.Action):
                if last_function_block is not None:
                    last_function_block.actions.append(
                        ActionSummary.from_action(
                            item,
                            source_code=get_code_by_meta(item.meta)
                        )
                    )
        return result


@dataclass
class LinkableItems:
    """A summary of linkable (located) declarations."""
    input: List[DeclarationSummary] = field(default_factory=list)
    output: List[DeclarationSummary] = field(default_factory=list)
    memory: List[DeclarationSummary] = field(default_factory=list)


def get_linkable_declarations(
    declarations: Iterable[DeclarationSummary],
) -> LinkableItems:
    """
    Get all located/linkable declarations.
    """
    linkable = LinkableItems()
    for decl in declarations:
        linkable_list = getattr(linkable, decl.location_type or "", None)
        if linkable_list is not None:
            linkable_list.append(decl)
    return linkable
