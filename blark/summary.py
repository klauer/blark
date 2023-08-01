from __future__ import annotations

import collections
import logging
import pathlib
import textwrap
from dataclasses import dataclass, field, fields, is_dataclass
from typing import (TYPE_CHECKING, Any, Dict, Generator, Iterable, List,
                    Optional, Tuple, Union)

import lark

from . import transform as tf
from .typing import Literal
from .util import Identifier, SourceType

LocationType = Literal["input", "output", "memory"]

if TYPE_CHECKING:
    from .parse import ParseResult


logger = logging.getLogger(__name__)


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
        result = [
            f"<{item.__class__.__name__}>"
        ]
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
    filename: Optional[pathlib.Path]
    meta: Optional[tf.Meta] = field(repr=False)

    def __str__(self) -> str:
        return text_outline(self) or ""

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
    """Summary representation of a single declaration."""
    name: str
    item: Union[tf.Declaration, tf.GlobalVariableDeclaration, tf.VariableInitDeclaration]
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
        item: Union[tf.InitDeclaration, tf.StructureElementDeclaration],
        parent: Optional[
            Union[tf.Function, tf.Method, tf.FunctionBlock, tf.StructureTypeDeclaration]
        ] = None,
        block_header: str = "unknown",
        filename: Optional[pathlib.Path] = None,
    ) -> Dict[str, DeclarationSummary]:
        result = {}
        if isinstance(item, tf.StructureElementDeclaration):
            result[item.name] = DeclarationSummary(
                name=str(item.name),
                item=item,
                location=item.location.name if item.location else None,
                block=block_header,
                type=item.init.full_type_name,
                base_type=item.init.base_type_name,
                value=str(item.init.value),
                parent=parent.name if parent is not None else "",
                filename=filename,
                **Summary.get_meta_kwargs(item.meta),
            )
        elif isinstance(
            item,
            (
                tf.FunctionBlockNameDeclaration,
                tf.FunctionBlockInvocationDeclaration,
            ),
        ):
            for var in item.variables:
                result[var] = DeclarationSummary(
                    name=str(var),
                    item=item,
                    location=None,
                    block=block_header,
                    type=item.init.full_type_name,
                    base_type=item.init.base_type_name,
                    value=str(item.init.value),
                    parent=parent.name if parent is not None else "",
                    filename=filename,
                    **Summary.get_meta_kwargs(item.meta),
                )
        elif isinstance(item, tf.InitDeclaration):
            for var in item.variables:
                result[var.name] = DeclarationSummary(
                    name=str(var.name),
                    item=item,
                    location=str(var.location).replace("AT ", "") if var.location else None,
                    block=block_header,
                    type=item.init.full_type_name,
                    base_type=item.init.base_type_name,
                    value=str(item.init.value),
                    parent=parent.name if parent is not None else "",
                    filename=filename,
                    **Summary.get_meta_kwargs(item.meta),
                )
        elif isinstance(item, tf.ExternalVariableDeclaration):
            location = str(getattr(item.spec, "location", None))
            init = str(getattr(item.spec, "init", None))
            type_ = getattr(init, "full_type_name", str(item.spec))
            base_type = getattr(init, "base_type_name", str(item.spec))
            result[item.name] = DeclarationSummary(
                name=str(item.name),
                item=item,
                location=location,
                block=block_header,
                type=type_,
                base_type=base_type,
                value="None",  # TODO
                parent=parent.name if parent is not None else "",
                filename=filename,
                **Summary.get_meta_kwargs(item.meta),
            )
        else:
            raise NotImplementedError(f"TODO: {type(item)}")

        return result

    @classmethod
    def from_global_variable(
        cls,
        item: tf.GlobalVariableDeclaration,
        parent: Optional[tf.GlobalVariableDeclarations] = None,
        block_header: str = "VAR_GLOBAL",
        filename: Optional[pathlib.Path] = None,
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
                filename=filename,
                **Summary.get_meta_kwargs(item.meta),
            )
        return result

    @classmethod
    def from_block(
        cls,
        block: tf.VariableDeclarationBlock,
        parent: Union[tf.Function, tf.Method, tf.FunctionBlock],
        filename: Optional[pathlib.Path] = None,
    ) -> Dict[str, DeclarationSummary]:
        result = {}
        for decl in block.items:
            result.update(
                cls.from_declaration(decl, parent=parent, block_header=block.block_header,
                                     filename=filename)
            )
        return result


@dataclass
class ActionSummary(Summary):
    """Summary representation of a single action."""
    name: str
    item: tf.Action
    source_code: str
    implementation: Optional[tf.StatementList] = None

    def __getitem__(self, key: str):
        raise KeyError(f"{key}: Actions do not contain declarations")

    @classmethod
    def from_statement_list(
        cls,
        name: str,
        statements: tf.StatementList,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> ActionSummary:
        if source_code is None:
            source_code = str(statements)

        return ActionSummary(
            name=name,
            item=tf.Action(
                name=lark.Token(name, name),
                body=statements,
                meta=statements.meta,
            ),
            source_code=source_code,
            filename=filename,
            implementation=statements,  # TODO: this is no good
            **Summary.get_meta_kwargs(statements.meta),
        )

    @classmethod
    def from_action(
        cls,
        action: tf.Action,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> ActionSummary:
        if source_code is None:
            source_code = str(action)

        return ActionSummary(
            name=str(action.name),
            item=action,
            source_code=source_code,
            filename=filename,
            **Summary.get_meta_kwargs(action.meta),
        )


@dataclass
class MethodSummary(Summary):
    """Summary representation of a single method."""
    name: str
    item: tf.Method
    return_type: Optional[str]
    source_code: str
    implementation: Optional[tf.StatementList] = None
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
    def from_method(
        cls,
        method: tf.Method,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> MethodSummary:
        if source_code is None:
            source_code = str(method)

        summary = MethodSummary(
            name=method.name,
            item=method,
            return_type=str(method.return_type) if method.return_type else None,
            source_code=source_code,
            filename=filename,
            **Summary.get_meta_kwargs(method.meta),
        )
        for decl in method.declarations:
            summary.declarations.update(
                DeclarationSummary.from_block(decl, parent=method, filename=filename)
            )

        return summary


@dataclass
class PropertyGetSetSummary(Summary):
    name: str
    item: tf.Property
    source_code: str
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    implementation: Optional[tf.StatementList] = None

    def __getitem__(self, key: str) -> DeclarationSummary:
        return self.declarations[key]


@dataclass
class PropertySummary(Summary):
    """Summary representation of a single property."""
    name: str
    getter: PropertyGetSetSummary
    setter: PropertyGetSetSummary
    source_code: str
    # implementation: Optional[tf.StatementList] = None

    def __getitem__(self, key: str):
        if key == "get":
            return self.getter
        if key == "set":
            return self.setter
        raise KeyError(f"{key}: Properties do not contain declarations")

    @classmethod
    def from_property(
        cls,
        property: tf.Property,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> PropertySummary:
        if source_code is None:
            source_code = str(property)

        # TODO: this is broken at the moment
        return PropertySummary(
            name=str(property.name),
            getter=PropertyGetSetSummary(
                name=str(property.name),
                item=property,
                source_code=source_code,
                filename=filename,
                **Summary.get_meta_kwargs(property.meta),
            ),
            setter=PropertyGetSetSummary(
                name=str(property.name),
                item=property,
                source_code=source_code,
                filename=filename,
                **Summary.get_meta_kwargs(property.meta),
            ),
            source_code=source_code,
            filename=filename,
            **Summary.get_meta_kwargs(property.meta),
        )


@dataclass
class FunctionSummary(Summary):
    """Summary representation of a single function."""
    name: str
    item: tf.Function
    return_type: Optional[str]
    source_code: str
    implementation: Optional[tf.StatementList] = None
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
        cls,
        func: tf.Function,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> FunctionSummary:
        if source_code is None:
            source_code = str(func)

        summary = FunctionSummary(
            name=func.name,
            item=func,
            return_type=str(func.return_type) if func.return_type else None,
            source_code=source_code,
            filename=filename,
            **Summary.get_meta_kwargs(func.meta),
        )

        for decl in func.declarations:
            summary.declarations.update(
                DeclarationSummary.from_block(decl, parent=func, filename=filename)
            )

        return summary


@dataclass
class FunctionBlockSummary(Summary):
    """Summary representation of a single function block."""
    name: str
    source_code: str
    item: tf.FunctionBlock
    extends: Optional[str]
    squashed: bool
    implementation: Optional[tf.StatementList] = None
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    actions: List[ActionSummary] = field(default_factory=list)
    methods: List[MethodSummary] = field(default_factory=list)
    properties: List[PropertySummary] = field(default_factory=list)

    def __getitem__(
        self, key: str
    ) -> Union[DeclarationSummary, MethodSummary, PropertySummary, ActionSummary]:
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
    def from_function_block(
        cls,
        fb: tf.FunctionBlock,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> FunctionBlockSummary:
        if source_code is None:
            source_code = str(fb)

        summary = FunctionBlockSummary(
            name=fb.name,
            item=fb,
            source_code=source_code,
            filename=filename,
            extends=fb.extends.name if fb.extends else None,
            squashed=False,
            **Summary.get_meta_kwargs(fb.meta),
        )

        for decl in fb.declarations:
            summary.declarations.update(
                DeclarationSummary.from_block(decl, parent=fb, filename=filename)
            )

        return summary

    def squash_base_extends(
        self, function_blocks: Dict[str, FunctionBlockSummary]
    ) -> FunctionBlockSummary:
        """Squash the "EXTENDS" function block into this one."""
        if self.extends is None:
            return self

        extends_from = function_blocks.get(str(self.extends), None)
        if extends_from is None:
            return self

        if extends_from.extends:
            extends_from = extends_from.squash_base_extends(function_blocks)

        declarations = dict(extends_from.declarations)
        declarations.update(self.declarations)
        actions = list(extends_from.actions) + self.actions
        methods = list(extends_from.methods) + self.methods
        properties = list(extends_from.properties) + self.properties
        return FunctionBlockSummary(
            name=self.name,
            comments=extends_from.comments + self.comments,
            pragmas=extends_from.pragmas + self.pragmas,
            meta=self.meta,
            filename=self.filename,
            source_code="\n\n".join((extends_from.source_code, self.source_code)),
            item=self.item,
            extends=self.extends,
            declarations=declarations,
            actions=actions,
            methods=methods,
            properties=properties,
            squashed=True,
        )


@dataclass
class InterfaceSummary(Summary):
    """Summary representation of an Interfae."""

    name: str
    source_code: str
    item: tf.Interface
    extends: Optional[str]
    squashed: bool
    declarations: Dict[str, DeclarationSummary] = field(default_factory=dict)
    methods: List[MethodSummary] = field(default_factory=list)
    properties: List[PropertySummary] = field(default_factory=list)
    # TwinCAT IDE doesn't allow for actions to be added to interfaces, it
    # seems. Overlap with methods?
    # actions: List[ActionSummary] = field(default_factory=list)

    def __getitem__(
        self, key: str
    ) -> Union[DeclarationSummary, MethodSummary, PropertySummary]:
        if key in self.declarations:
            return self.declarations[key]
        for item in self.methods + self.properties:
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
    def from_interface(
        cls,
        itf: tf.Interface,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> InterfaceSummary:
        if source_code is None:
            source_code = str(itf)

        summary = InterfaceSummary(
            name=itf.name,
            item=itf,
            source_code=source_code,
            filename=filename,
            extends=itf.extends.name if itf.extends else None,
            squashed=False,
            **Summary.get_meta_kwargs(itf.meta),
        )

        for decl in itf.declarations:
            summary.declarations.update(
                DeclarationSummary.from_block(decl, parent=itf, filename=filename)
            )

        return summary

    def squash_base_extends(
        self, interfaces: Dict[str, InterfaceSummary]
    ) -> InterfaceSummary:
        """Squash the "EXTENDS" INTERFACE into this one."""
        if self.extends is None:
            return self

        extends_from = interfaces.get(str(self.extends), None)
        if extends_from is None:
            return self

        if extends_from.extends:
            extends_from = extends_from.squash_base_extends(interfaces)

        declarations = dict(extends_from.declarations)
        declarations.update(self.declarations)
        methods = list(extends_from.methods) + self.methods
        properties = list(extends_from.properties) + self.properties
        return InterfaceSummary(
            name=self.name,
            comments=extends_from.comments + self.comments,
            pragmas=extends_from.pragmas + self.pragmas,
            meta=self.meta,
            filename=self.filename,
            source_code="\n\n".join((extends_from.source_code, self.source_code)),
            item=self.item,
            extends=self.extends,
            declarations=declarations,
            properties=properties,
            methods=methods,
            squashed=True,
        )


@dataclass
class DataTypeSummary(Summary):
    """Summary representation of a single data type."""
    # Note: structures only for now.
    name: str
    item: tf.TypeDeclarationItem
    source_code: str
    type: str
    extends: Optional[str]
    squashed: bool = False
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
        cls,
        dtype: tf.TypeDeclarationItem,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> DataTypeSummary:
        if source_code is None:
            source_code = str(dtype)

        if isinstance(dtype, tf.StructureTypeDeclaration):
            extends = dtype.extends.name if dtype.extends else None
        else:
            extends = None

        summary = cls(
            name=dtype.name,
            item=dtype,
            extends=extends,
            source_code=source_code,
            type=type(dtype).__name__,
            filename=filename,
            squashed=False,
            **Summary.get_meta_kwargs(dtype.meta),
        )

        if isinstance(dtype, tf.StructureTypeDeclaration):
            for decl in dtype.declarations:
                summary.declarations.update(
                    DeclarationSummary.from_declaration(
                        decl,
                        parent=dtype,
                        block_header="STRUCT",
                        filename=filename,
                    )
                )

        return summary

    def squash_base_extends(
        self, data_types: Dict[str, DataTypeSummary]
    ) -> DataTypeSummary:
        """Squash the "EXTENDS" function block into this one."""
        if self.extends is None:
            return self

        extends_from = data_types.get(str(self.extends), None)
        if extends_from is None:
            return self

        if extends_from.extends:
            extends_from = extends_from.squash_base_extends(data_types)

        declarations = dict(extends_from.declarations)
        declarations.update(self.declarations)
        return DataTypeSummary(
            name=self.name,
            type=self.type,
            comments=extends_from.comments + self.comments,
            pragmas=extends_from.pragmas + self.pragmas,
            meta=self.meta,
            filename=self.filename,
            source_code="\n\n".join((extends_from.source_code, self.source_code)),
            item=self.item,
            extends=self.extends,
            declarations=declarations,
            squashed=True,
        )


@dataclass
class GlobalVariableSummary(Summary):
    """Summary representation of a VAR_GLOBAL block."""
    name: str
    item: tf.GlobalVariableDeclarations
    source_code: str
    type: str
    qualified_only: bool = False
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
        cls,
        decls: tf.GlobalVariableDeclarations,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> GlobalVariableSummary:
        if source_code is None:
            source_code = str(decls)

        summary = GlobalVariableSummary(
            name=decls.name or "(unknown)",
            item=decls,
            source_code=source_code,
            type=type(decls).__name__,
            filename=filename,
            qualified_only="qualified_only" in decls.attribute_pragmas,
            **Summary.get_meta_kwargs(decls.meta),
        )

        for decl in decls.items:
            summary.declarations.update(
                **DeclarationSummary.from_global_variable(
                    decl,
                    parent=summary,
                    block_header="VAR_GLOBAL",
                    filename=filename,
                )
            )

        return summary


@dataclass
class ProgramSummary(Summary):
    """Summary representation of a single program."""
    name: str
    source_code: str
    item: tf.Program
    implementation: Optional[tf.StatementList] = None
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
        cls,
        program: tf.Program,
        source_code: Optional[str] = None,
        filename: Optional[pathlib.Path] = None,
    ) -> ProgramSummary:
        if source_code is None:
            source_code = str(program)

        summary = ProgramSummary(
            name=program.name,
            item=program,
            source_code=source_code,
            filename=filename,
            **Summary.get_meta_kwargs(program.meta),
        )

        for decl in program.declarations:
            summary.declarations.update(
                DeclarationSummary.from_block(decl, parent=program, filename=filename)
            )

        return summary


def path_to_file_and_line(path: List[Summary]) -> List[Tuple[pathlib.Path, int]]:
    """Get the file/line number context for the summary items."""
    return [(part.filename, part.item.meta.line) for part in path]


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
    interfaces: Dict[str, InterfaceSummary] = field(default_factory=dict)

    def __str__(self):
        attr_to_header = {
            "data_types": "Data Types",
            "globals": "Global Variable Declarations",
            "interfaces": "Interface Declarations",
            "functions": "Functions",
            "function_blocks": "Function Blocks",
            "programs": "Programs",
        }
        summary_text = []
        for attr, header in attr_to_header.items():
            name_to_obj = getattr(self, attr)
            if name_to_obj:
                summary_text.extend(
                    [
                        header,
                        "-" * len(header),
                    ]
                )

                for name, obj in name_to_obj.items():
                    obj_info = "\n".join(
                        (
                            name,
                            "=" * len(name),
                            str(obj)
                        )
                    )
                    summary_text.append(textwrap.indent(obj_info, " " * 4))

        return "\n".join(summary_text)

    def find(self, name: str) -> Optional[Summary]:
        """Find a declaration or other item by its qualified name."""
        path = self.find_path(name)
        return path[-1] if path else None

    def find_path(self, name: str) -> Optional[List[Summary]]:
        """Given a qualified variable name, find its Declaration."""
        parts = collections.deque(name.split("."))
        if len(parts) <= 1:
            item = self.get_item_by_name(name)
            return [item] if item is not None else None

        variable_name = parts.pop()
        parent = None
        path = []
        while parts:
            part = parts.popleft()
            if "[" in part:  # ]
                part = part.split("[")[0]  # ]

            try:
                if parent is None:
                    parent = self.get_item_by_name(part)
                else:
                    part_obj = parent[part]
                    path.append(part_obj)
                    part_type = str(part_obj.base_type)
                    parent = self.get_item_by_name(part_type)
            except KeyError:
                return

        if parent is None:
            return

        try:
            path.append(parent[variable_name])
        except KeyError:
            # Is it better to give a partial path or no path at all?
            ...

        return path

    def find_code_object_by_dotted_name(self, name: str) -> Optional[Summary]:
        """Given a qualified code object name, find its Summary object(s)."""
        # TODO: These functions are a bit of a mess and confuse even me
        # The intent with *this* flavor of 'find' is to take in things like:
        # FB_Block.ActionName
        # FB_Block.PropertyName.get
        # FB_Block.PropertyName.set
        parts = Identifier.from_string(name).parts
        obj = self.get_item_by_name(parts[0])
        if len(parts) == 1:
            return obj

        for remainder in parts[1:]:
            obj = obj[remainder]
        return obj

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
        self.interfaces.update(other.interfaces)

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

        self.squash()

    @staticmethod
    def from_parse_results(
        all_parsed_items: list[ParseResult],
        squash: bool = True,
    ) -> CodeSummary:
        result = CodeSummary()

        def get_code_by_meta(parsed: ParseResult, meta: Optional[tf.Meta]) -> str:
            if meta is None or meta.line is None or meta.end_line is None:
                return ""

            transformed = parsed.transform()
            return "\n".join(
                transformed.range_from_file_lines(meta.line, meta.end_line)
            )

        def add_implementation(parsed: ParseResult, impl: tf.StatementList):
            assert parsed.identifier is not None
            identifier = Identifier.from_string(parsed.identifier)
            assert identifier.decl_impl == "implementation"

            match = result.find_code_object_by_dotted_name(identifier.dotted_name)
            if match is None:
                raise RuntimeError(
                    f"Implementation without previous declaration? "
                    f"{parsed.filename} {parsed.identifier}"
                )
            # if len(matches) > 1:
            #     raise RuntimeError(
            #         f"Multiple matches for implementation? "
            #         f"{parsed.filename} {parsed.identifier}"
            #     )

            if match.implementation is not None:
                raise RuntimeError(
                    f"Implementation specified twice for {parsed.filename} "
                    f"{parsed.identifier}"
                )

            if isinstance(match, PropertyGetSetSummary):
                ...

            match.implementation = impl

        context = []

        def clear_context():
            context.clear()

        def new_context(summary: Summary):
            context[:] = [summary]

        def push_context(summary: Summary):
            context.append(summary)

        def get_pou_context() -> Union[
            ProgramSummary, FunctionBlockSummary, InterfaceSummary
        ]:
            for item in reversed(context):
                if isinstance(
                    item, (ProgramSummary, FunctionBlockSummary, InterfaceSummary)
                ):
                    return item

            raise ValueError(
                "Expected to parse a POU prior to this but none were in the context "
                "list.  Code summaries of PROPERTY objects, for example, require "
                "that a FUNCTION_BLOCK, INTERFACE, or PROGRAM be parsed previously."
            )

        for parsed in all_parsed_items:
            transformed = parsed.transform()
            identifier = Identifier.from_string(parsed.identifier) if parsed.identifier else None
            for item in transformed.items:
                if isinstance(item, tf.FunctionBlock):
                    summary = FunctionBlockSummary.from_function_block(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    result.function_blocks[item.name] = summary
                    new_context(summary)
                elif isinstance(item, tf.Function):
                    summary = FunctionSummary.from_function(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    result.functions[item.name] = summary
                    new_context(summary)
                elif isinstance(item, tf.DataTypeDeclaration):
                    if isinstance(item.declaration, tf.StructureTypeDeclaration):
                        summary = DataTypeSummary.from_data_type(
                            item.declaration,
                            source_code=get_code_by_meta(parsed, item.declaration.meta),
                            filename=parsed.filename,
                        )
                        result.data_types[item.declaration.name] = summary
                    clear_context()
                elif isinstance(item, tf.Method):
                    pou = get_pou_context()
                    summary = MethodSummary.from_method(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    pou.methods.append(summary)
                    push_context(summary)
                elif isinstance(item, tf.Action):
                    pou = get_pou_context()
                    summary = ActionSummary.from_action(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    pou.actions.append(summary)
                    push_context(summary)
                elif isinstance(item, tf.Property):
                    pou = get_pou_context()
                    summary = PropertySummary.from_property(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    pou.properties.append(summary)
                    push_context(summary)
                elif isinstance(item, tf.GlobalVariableDeclarations):
                    clear_context()
                    summary = GlobalVariableSummary.from_globals(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    result.globals[item.name] = summary
                    # for global_var in summary.declarations.values():
                    #     if not qualified_only:
                    #         result.globals[global_var.name] = summary
                    #     result.globals[global_var.qualified_name] = summary

                elif isinstance(item, tf.Program):
                    summary = ProgramSummary.from_program(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    result.programs[item.name] = summary
                    new_context(summary)
                elif isinstance(item, tf.Interface):
                    summary = InterfaceSummary.from_interface(
                        item,
                        source_code=get_code_by_meta(parsed, item.meta),
                        filename=parsed.filename,
                    )
                    result.interfaces[item.name] = summary
                    new_context(summary)
                elif isinstance(item, tf.StatementList):
                    if parsed.item.type != SourceType.action:
                        add_implementation(parsed, item)
                    else:
                        assert identifier is not None
                        # Special-case: actions only have implementations
                        parent_identifier = Identifier(parts=identifier.parts[:-1])
                        parent, = result.find_path(parent_identifier.dotted_name)
                        assert isinstance(parent, (FunctionBlockSummary, ProgramSummary))
                        action = ActionSummary.from_statement_list(
                            name=identifier.parts[-1],
                            statements=item,
                            source_code=parsed.source_code,  # TODO above?
                            filename=parsed.filename,
                        )
                        parent.actions.append(action)
                else:
                    raise ValueError(type(item))
                    logger.warning("Unhandled: %s", type(item))

        if squash:
            result.squash()

        return result

    def squash(self) -> None:
        """Squash derived interfaces/etc to include base summaries."""
        for name, item in list(self.function_blocks.items()):
            if item.extends and not item.squashed:
                self.function_blocks[name] = item.squash_base_extends(
                    self.function_blocks
                )

        for name, item in list(self.data_types.items()):
            if item.extends and not item.squashed:
                self.data_types[name] = item.squash_base_extends(
                    self.data_types
                )

        for name, item in list(self.interfaces.items()):
            if item.extends and not item.squashed:
                self.interfaces[name] = item.squash_base_extends(
                    self.interfaces
                )


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
