from __future__ import annotations

import textwrap
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

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
    item: Union[tf.Declaration, tf.GlobalVariableDeclaration]
    parent: Optional[str]
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
        for var in item.variables:
            name = getattr(var, "name", var)
            location = getattr(var, "location", None)
            result[name] = DeclarationSummary(
                name=str(name),
                item=item,
                location=str(location).replace("AT ", "") if location else None,
                block=block_header,
                type=item.init.full_type_name,
                base_type=item.init.base_type_name,
                value=str(item.init.value),
                parent=parent.name if parent is not None else "",
                **Summary.get_meta_kwargs(item.meta),
            )
        return result

    @classmethod
    def from_global_variable(
        cls,
        item: tf.GlobalVariableDeclaration,
        parent: Optional[tf.GlobalVariableDeclarations] = None,
        block_header: str = "VAR_GLOBAL",
    ) -> Dict[str, DeclarationSummary]:
        result = {}
        location = (str(item.spec.location or "").replace("AT ", "")) or None

        for var in item.spec.variables:
            name = getattr(var, "name", var)
            result[name] = DeclarationSummary(
                name=str(name),
                item=item,
                location=location,
                block=block_header,
                type=item.full_type_name,
                base_type=item.base_type_name,
                value=str(item.init.value),
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
    item: tf.Action
    source_code: str

    def __getitem__(self, key: str):
        raise KeyError(f"{key}: Actions do not contain declarations")

    @classmethod
    def from_action(cls, action: tf.Action, source_code: Optional[str] = None) -> ActionSummary:
        if source_code is None:
            source_code = str(action)

        return ActionSummary(
            name=str(action.name),
            item=action,
            source_code=source_code,
            **Summary.get_meta_kwargs(action.meta),
        )


@dataclass
class MethodSummary(Summary):
    """Summary representation of a single method."""
    name: str
    item: tf.Method
    return_type: Optional[str]
    source_code: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    def __getitem__(self, key: str) -> DeclarationSummary:
        return self.declarations[key]

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
            item=method,
            return_type=str(method.return_type) if method.return_type else None,
            source_code=source_code,
            **Summary.get_meta_kwargs(method.meta),
        )
        for decl in method.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl, parent=method))

        return summary


@dataclass
class PropertySummary(Summary):
    """Summary representation of a single property."""
    name: str
    item: tf.Property
    source_code: str

    def __getitem__(self, key: str):
        raise KeyError(f"{key}: Properties do not contain declarations")

    @classmethod
    def from_property(
        cls, property: tf.Property, source_code: Optional[str] = None
    ) -> PropertySummary:
        if source_code is None:
            source_code = str(property)

        return PropertySummary(
            name=str(property.name),
            item=property,
            source_code=source_code,
            **Summary.get_meta_kwargs(property.meta),
        )


@dataclass
class FunctionSummary(Summary):
    """Summary representation of a single function."""
    name: str
    item: tf.Function
    return_type: Optional[str]
    source_code: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    def __getitem__(self, key: str) -> DeclarationSummary:
        return self.declarations[key]

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
            item=func,
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
    item: tf.FunctionBlock
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    actions: List[ActionSummary] = field(default_factory=list)
    methods: List[MethodSummary] = field(default_factory=list)

    def __getitem__(self, key: str) -> DeclarationSummary:
        if key in self.declarations:
            return self.declarations[key]
        for item in self.actions + self.methods:
            if item.name == key:
                return item
        raise KeyError(key)

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
            item=fb,
            source_code=source_code,
            **Summary.get_meta_kwargs(fb.meta),
        )

        for decl in fb.declarations:
            summary.declarations.update(DeclarationSummary.from_block(decl, parent=fb))

        return summary


@dataclass
class DataTypeSummary(Summary):
    """Summary representation of a single data type."""
    # Note: structures only for now.
    name: str
    item: tf.TypeDeclarationItem
    source_code: str
    type: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    def __getitem__(self, key: str) -> DeclarationSummary:
        return self.declarations[key]

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
            item=dtype,
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
class GlobalVariableSummary(Summary):
    """Summary representation of a VAR_GLOBAL block."""
    name: str
    item: tf.GlobalVariableDeclarations
    source_code: str
    type: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)

    def __getitem__(self, key: str) -> DeclarationSummary:
        return self.declarations[key]

    @property
    def declarations_by_block(self) -> Dict[str, Dict[str, DeclarationSummary]]:
        return {
            "VAR_GLOBAL": self.declarations
        }

    @classmethod
    def from_globals(
        cls, decls: tf.GlobalVariableDeclarations, source_code: Optional[str] = None
    ) -> GlobalVariableSummary:
        if source_code is None:
            source_code = str(decls)

        summary = GlobalVariableSummary(
            name=decls.name or "(unknown)",
            item=decls,
            source_code=source_code,
            type=type(decls).__name__,
            **Summary.get_meta_kwargs(decls.meta),
        )

        for decl in decls.items:
            summary.declarations.update(
                **DeclarationSummary.from_global_variable(
                    decl,
                    parent=summary,
                    block_header="VAR_GLOBAL"
                )
            )

        return summary


@dataclass
class ProgramSummary(Summary):
    """Summary representation of a single program."""
    name: str
    source_code: str
    item: tf.Program
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    actions: List[ActionSummary] = field(default_factory=list)
    methods: List[MethodSummary] = field(default_factory=list)
    properties: List[PropertySummary] = field(default_factory=list)

    def __getitem__(self, key: str) -> DeclarationSummary:
        if key in self.declarations:
            return self.declarations[key]
        for item in self.actions + self.methods + self.properties:
            if item.name == key:
                return item
        raise KeyError(key)

    @property
    def declarations_by_block(self) -> Dict[str, Dict[str, DeclarationSummary]]:
        result = {}
        for decl in self.declarations.values():
            result.setdefault(decl.block, {})[decl.name] = decl
        return result

    @classmethod
    def from_program(
        cls, program: tf.Program, source_code: Optional[str] = None
    ) -> ProgramSummary:
        if source_code is None:
            source_code = str(program)

        summary = ProgramSummary(
            name=program.name,
            item=program,
            source_code=source_code,
            **Summary.get_meta_kwargs(program.meta),
        )

        for decl in program.declarations:
            summary.declarations.update(
                DeclarationSummary.from_block(decl, parent=program)
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
    programs: Dict[str, ProgramSummary] = field(default_factory=dict)
    globals: Dict[str, GlobalVariableSummary] = field(default_factory=dict)

    def __str__(self):
        return "\n".join(
            f"{name}:\n{fb}"
            for name, fb in self.function_blocks.items()
        )

    def get_all_items_by_name(self, name: str) -> Generator:
        """Get any code item (function, data type, global variable, etc.) by name."""
        for dct in (
            self.globals,
            self.programs,
            self.functions,
            self.function_blocks,
            self.data_types,
        ):
            # Very inefficient, be warned
            try:
                yield dct[name]
            except KeyError:
                ...

    def get_item_by_name(self, name: str) -> Optional[Any]:
        """Get any code item (function, data type, global variable, etc.) by name."""
        try:
            return next(self.get_all_items_by_name(name))
        except StopIteration:
            return None

    def append(self, other: CodeSummary, namespace: Optional[str] = None):
        """
        In-place add code summary information from another instance.

        New entries take precedence over old ones.
        """

        self.functions.update(other.functions)
        self.function_blocks.update(other.function_blocks)
        self.data_types.update(other.data_types)
        self.globals.update(other.globals)
        self.programs.update(other.programs)

        if namespace:
            # LCLS_General.GVL_Logger and GVL_Logger are equally valid
            for name, item in other.functions.items():
                self.functions[f"{namespace}.{name}"] = item
            for name, item in other.function_blocks.items():
                self.function_blocks[f"{namespace}.{name}"] = item
            for name, item in other.data_types.items():
                self.data_types[f"{namespace}.{name}"] = item
            for name, item in other.globals.items():
                self.globals[f"{namespace}.{name}"] = item
            # for name, item in other.programs.items():
            #     self.programs[f"{namespace}.{name}"] = item

    @staticmethod
    def from_source(code: tf.SourceCode) -> CodeSummary:
        result = CodeSummary()
        code_by_lines = [""] + code.raw_source.splitlines()
        items = code.items

        def get_code_by_meta(meta: Optional[tf.Meta]) -> str:
            if not meta:
                return ""
            return "\n".join(code_by_lines[meta.line:meta.end_line + 1])

        last_parent = None
        for item in items:
            if isinstance(item, tf.FunctionBlock):
                summary = FunctionBlockSummary.from_function_block(
                    item,
                    source_code=get_code_by_meta(item.meta)
                )
                result.function_blocks[item.name] = summary
                last_parent = summary
            elif isinstance(item, tf.Function):
                summary = FunctionSummary.from_function(
                    item,
                    source_code=get_code_by_meta(item.meta)
                )
                result.functions[item.name] = summary
                last_parent = None
            elif isinstance(item, tf.DataTypeDeclaration):
                if isinstance(item.declaration, tf.StructureTypeDeclaration):
                    summary = DataTypeSummary.from_data_type(
                        item.declaration,
                        source_code=get_code_by_meta(item.declaration.meta)
                    )
                    result.data_types[item.declaration.name] = summary
                last_parent = None
            elif isinstance(item, tf.Method):
                if last_parent is not None:
                    last_parent.methods.append(
                        MethodSummary.from_method(
                            item,
                            source_code=get_code_by_meta(item.meta)
                        )
                    )
            elif isinstance(item, tf.Action):
                if last_parent is not None:
                    last_parent.actions.append(
                        ActionSummary.from_action(
                            item,
                            source_code=get_code_by_meta(item.meta)
                        )
                    )
            elif isinstance(item, tf.Property):
                if last_parent is not None and isinstance(last_parent, ProgramSummary):
                    last_parent.properties.append(
                        PropertySummary.from_property(
                            item,
                            source_code=get_code_by_meta(item.meta)
                        )
                    )
            elif isinstance(item, tf.GlobalVariableDeclarations):
                qualified_only = "qualified_only" in item.attribute_pragmas
                summary = GlobalVariableSummary.from_globals(
                    item,
                    source_code=get_code_by_meta(item.meta)
                )

                for global_var in summary.declarations.values():
                    if not qualified_only:
                        result.globals[global_var.name] = summary
                    result.globals[global_var.qualified_name] = summary

            elif isinstance(item, tf.Program):
                summary = ProgramSummary.from_program(
                    item,
                    source_code=get_code_by_meta(item.meta)
                )
                result.programs[item.name] = summary
                last_parent = summary

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
