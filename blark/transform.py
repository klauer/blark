from __future__ import annotations

import dataclasses
import enum
import functools
import pathlib
import textwrap
import typing
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import (Any, Callable, ClassVar, Dict, Generator, List, Optional,
                    Tuple, Type, TypeVar, Union)

import lark

from .util import AnyPath

T = TypeVar("T")

try:
    # NOTE: apischema is an optional requirement; this should work regardless.
    import apischema

    from .apischema_compat import as_tagged_union
except ImportError:
    apischema = None

    def as_tagged_union(cls: Type[T]) -> Type[T]:
        """No-operation stand-in for when apischema is not available."""
        return cls


_rule_to_class: Dict[str, type] = {}
_class_handlers = {}
_comment_consumers = []

INDENT = "    "  # TODO: make it configurable


def multiline_code_block(block: str) -> str:
    """Multiline code block with lax beginning/end newlines."""
    return textwrap.dedent(block.strip("\n")).rstrip()


def join_if(value1: Optional[Any], delimiter: str, value2: Optional[Any]) -> str:
    """'{value1}{delimiter}{value2} if value1 and value2, otherwise just {value1} or {value2}."""
    return delimiter.join(
        str(value) for value in (value1, value2)
        if value is not None
    )


def indent_if(value: Optional[Any], prefix: str = INDENT) -> Optional[str]:
    """Stringified and indented {value} if not None."""
    if value is not None:
        return textwrap.indent(str(value), prefix)
    return None


def indent(value: Any, prefix: str = INDENT) -> str:
    """Stringified and indented {value}."""
    return textwrap.indent(str(value), prefix)


def _commented(meta: Meta, item: Any, indent: str = "", suffix="") -> str:
    comments = getattr(meta, "comments", None)
    if not comments:
        return f"{indent}{item}{suffix}"

    block = "\n".join((*comments, f"{item}{suffix}"))
    return textwrap.indent(block, prefix=indent)


def _add_comments_to_return_value(func):
    @functools.wraps(func)
    def wrapped(self):
        return _commented(
            self.meta,
            func(self)
        )

    if getattr(func, "_comment_wrapped", None):
        return func  # pragma: no cover

    wrapped._comment_wrapped = True
    return wrapped


def _get_default_instantiator(cls: Type[T]):
    def instantiator(*args) -> T:
        return cls(*args)

    return instantiator


def _rule_handler(
    *rules: str,
    comments: bool = False
) -> Callable[[Type[T]], Type[T]]:
    """Decorator - the wrapped class will handle the provided rules."""
    def wrapper(cls: Type[T]) -> Type[T]:
        if not hasattr(cls, "from_lark"):
            cls.from_lark = _get_default_instantiator(cls)

        for rule in rules:
            handler = _rule_to_class.get(rule, None)
            if handler is not None:
                raise ValueError(
                    f"Handler already specified for: {rule} ({handler})"
                )  # pragma: no cover

            _rule_to_class[rule] = cls
            _class_handlers[rule] = cls.from_lark

        if comments:
            # Mark ``cls`` as one that consumes comments when stringifying code.
            _comment_consumers.append(cls)
            cls.__str__ = _add_comments_to_return_value(cls.__str__)

        return cls

    return wrapper


@dataclasses.dataclass
class Meta:
    empty: bool = True
    column: Optional[int] = None
    comments: List[lark.Token] = dataclasses.field(default_factory=list)
    container_column: Optional[int] = None
    container_end_column: Optional[int] = None
    container_end_line: Optional[int] = None
    container_line: Optional[int] = None
    end_column: Optional[int] = None
    end_line: Optional[int] = None
    end_pos: Optional[int] = None
    line: Optional[int] = None
    start_pos: Optional[int] = None

    @staticmethod
    def from_lark(lark_meta: lark.tree.Meta) -> Meta:
        return Meta(
            empty=lark_meta.empty,
            column=getattr(lark_meta, "column", None),
            comments=getattr(lark_meta, "comments", []),
            container_column=getattr(lark_meta, "container_column", None),
            container_end_column=getattr(lark_meta, "container_end_column", None),
            container_end_line=getattr(lark_meta, "container_end_line", None),
            container_line=getattr(lark_meta, "container_line", None),
            end_column=getattr(lark_meta, "end_column", None),
            end_line=getattr(lark_meta, "end_line", None),
            end_pos=getattr(lark_meta, "end_pos", None),
            line=getattr(lark_meta, "line", None),
            start_pos=getattr(lark_meta, "start_pos", None),
        )

    def get_comments_and_pragmas(self) -> Tuple[List[lark.Token], List[lark.Token]]:
        """
        Split the contained comments into comments/pragmas.

        Returns
        -------
        comments : List[lark.Token]
        pragmas : List[lark.Token]
        """
        if not self.comments:
            return [], []

        pragmas: List[lark.Token] = []
        comments: List[lark.Token] = []
        by_type = {
            "SINGLE_LINE_COMMENT": comments,
            "MULTI_LINE_COMMENT": comments,
            "PRAGMA": pragmas,
        }

        for comment in self.comments:
            by_type[comment.type].append(comment)

        return comments, pragmas

    @property
    def attribute_pragmas(self) -> List[str]:
        """Get {attribute ...} pragmas associated with this code block."""

        _, pragmas = self.get_comments_and_pragmas()
        attributes = []
        for pragma in pragmas:
            # TODO: better pragma parsing; it's its own grammar
            if pragma.startswith("{attribute "):  # }
                attributes.append(pragma.split(" ")[1].strip(" }'"))
        return attributes


def meta_field():
    """Create the Meta field for the dataclass magic."""
    return dataclasses.field(default=None, repr=False, compare=False)


class _FlagHelper:
    @classmethod
    def from_lark(cls, token: lark.Token, *tokens: lark.Token):
        result = cls[token.lower()]
        for token in tokens:
            result |= cls[token.lower()]
        return result

    def __str__(self):
        return " ".join(
            option.name.upper()
            for option in type(self)
            if option in self
        )


@_rule_handler("variable_attributes")
class VariableAttributes(_FlagHelper, enum.IntFlag):
    constant = 0b0000_0001
    retain = 0b0000_0010
    non_retain = 0b0000_0100
    persistent = 0b0000_1000


@_rule_handler("global_variable_attributes")
class GlobalVariableAttributes(_FlagHelper, enum.IntFlag):
    constant = 0b0000_0001
    retain = 0b0000_0010
    non_retain = 0b0000_0100
    persistent = 0b0000_1000
    internal = 0b0001_0000


@_rule_handler(
    "access_specifier",
)
class AccessSpecifier(_FlagHelper, enum.IntFlag):
    public = 0b0000_0001
    private = 0b0000_0010
    abstract = 0b0000_0100
    protected = 0b0000_1000
    internal = 0b0001_0000
    final = 0b010_0000


@dataclass
@as_tagged_union
class Expression:
    ...


@as_tagged_union
class Literal(Expression):
    """Literal value."""


@dataclass
@_rule_handler("integer_literal")
class Integer(Literal):
    """Integer literal value."""

    value: lark.Token
    type_name: Optional[lark.Token] = None
    base: ClassVar[int] = 10
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        type_name: Optional[lark.Token],
        value: Union[Integer, lark.Token],
        *,
        base: int = 10,
    ) -> Integer:
        if isinstance(value, Integer):
            # Adding type_name information; wrap Integer
            value.type_name = type_name
            return value
        cls = _base_to_integer_class[base]
        return cls(
            type_name=type_name,
            value=value,
        )

    def __str__(self) -> str:
        value = f"{self.base}#{self.value}" if self.base != 10 else str(self.value)
        if self.type_name:
            return f"{self.type_name}#{value}"
        return value


@dataclass
@_rule_handler("binary_integer")
class BinaryInteger(Integer):
    base: ClassVar[int] = 2


@dataclass
@_rule_handler("octal_integer")
class OctalInteger(Integer):
    base: ClassVar[int] = 8


@dataclass
@_rule_handler("hex_integer")
class HexInteger(Integer):
    base: ClassVar[int] = 16


_base_to_integer_class: Dict[int, Type[Integer]] = {
    2: BinaryInteger,
    8: OctalInteger,
    10: Integer,
    16: HexInteger,
}


@dataclass
@_rule_handler("real_literal")
class Real(Literal):
    """Floating point (real) literal value."""

    value: lark.Token
    type_name: Optional[lark.Token] = None
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(type_name: Optional[lark.Token], value: lark.Token) -> Real:
        return Real(type_name=type_name, value=value)

    def __str__(self) -> str:
        if self.type_name:
            return f"{self.type_name}#{self.value}"
        return str(self.value)


@dataclass
@_rule_handler("bit_string_literal")
class BitString(Literal):
    """Bit string literal value."""

    type_name: Optional[lark.Token]
    value: lark.Token
    base: ClassVar[int] = 10
    meta: Optional[Meta] = meta_field()

    @classmethod
    def from_lark(cls, type_name: Optional[lark.Token], value: lark.Token):
        return cls(type_name, value)

    def __str__(self) -> str:
        value = f"{self.base}#{self.value}" if self.base != 10 else str(self.value)
        if self.type_name:
            return f"{self.type_name}#{value}"
        return value


@dataclass
@_rule_handler("binary_bit_string_literal")
class BinaryBitString(BitString):
    """Binary bit string literal value."""
    base: ClassVar[int] = 2
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler("octal_bit_string_literal")
class OctalBitString(BitString):
    """Octal bit string literal value."""
    base: ClassVar[int] = 8
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler("hex_bit_string_literal")
class HexBitString(BitString):
    """Hex bit string literal value."""
    base: ClassVar[int] = 16
    meta: Optional[Meta] = meta_field()


@dataclass
class Boolean(Literal):
    """Boolean literal value."""
    value: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        value = self.value.lower() in ("1", "true")
        return "TRUE" if value else "FALSE"


@dataclass
@_rule_handler("duration")
class Duration(Literal):
    """Duration literal value."""

    days: Optional[lark.Token] = None
    hours: Optional[lark.Token] = None
    minutes: Optional[lark.Token] = None
    seconds: Optional[lark.Token] = None
    milliseconds: Optional[lark.Token] = None
    negative: bool = False
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(minus: Optional[lark.Token], interval: lark.Tree) -> Duration:
        kwargs = typing.cast(
            Dict[str, lark.Token],
            {
                tree.data: tree.children[0]
                for tree in interval.iter_subtrees()
            }
        )
        return Duration(**kwargs, negative=minus is not None, meta=None)

    @property
    def value(self) -> str:
        """The duration value."""
        prefix = "-" if self.negative else ""
        return prefix + "".join(
            f"{value}{suffix}"
            for value, suffix in (
                (self.days, "D"),
                (self.hours, "H"),
                (self.minutes, "M"),
                (self.seconds, "S"),
                (self.milliseconds, "MS"),
            )
            if value is not None
        )

    def __str__(self):
        return f"TIME#{self.value}"


@dataclass
@_rule_handler("lduration")
class Lduration(Literal):
    """Long duration literal value."""

    days: Optional[lark.Token] = None
    hours: Optional[lark.Token] = None
    minutes: Optional[lark.Token] = None
    seconds: Optional[lark.Token] = None
    milliseconds: Optional[lark.Token] = None
    microseconds: Optional[lark.Token] = None
    nanoseconds: Optional[lark.Token] = None
    negative: bool = False
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(minus: Optional[lark.Token], interval: lark.Tree) -> Lduration:
        kwargs = typing.cast(
            Dict[str, lark.Token],
            {
                tree.data: tree.children[0]
                for tree in interval.iter_subtrees()
            }
        )
        return Lduration(**kwargs, negative=minus is not None, meta=None)

    @property
    def value(self) -> str:
        """The long duration value."""
        prefix = "-" if self.negative else ""
        return prefix + "".join(
            f"{value}{suffix}"
            for value, suffix in (
                (self.days, "D"),
                (self.hours, "H"),
                (self.minutes, "M"),
                (self.seconds, "S"),
                (self.milliseconds, "MS"),
                (self.microseconds, "US"),
                (self.nanoseconds, "NS"),
            )
            if value is not None
        )

    def __str__(self):
        return f"LTIME#{self.value}"


@dataclass
@_rule_handler("time_of_day")
class TimeOfDay(Literal):
    """Time of day literal value."""
    hour: lark.Token
    minute: lark.Token
    second: Optional[lark.Token] = None
    meta: Optional[Meta] = meta_field()

    @property
    def value(self) -> str:
        """The time of day value."""
        return join_if(f"{self.hour}:{self.minute}", ":", self.second)

    def __str__(self):
        return f"TIME_OF_DAY#{self.value}"


@dataclass
@_rule_handler("ltime_of_day")
class LtimeOfDay(Literal):
    """Long time of day literal value."""
    hour: lark.Token
    minute: lark.Token
    second: Optional[lark.Token] = None
    meta: Optional[Meta] = meta_field()

    @property
    def value(self) -> str:
        """The long time of day value."""
        return f"{self.hour}:{self.minute}:{self.second}"

    def __str__(self):
        return f"LTIME_OF_DAY#{self.value}"


@dataclass
@_rule_handler("date")
class Date(Literal):
    """Date literal value."""

    year: lark.Token
    month: lark.Token
    day: Optional[lark.Token]
    meta: Optional[Meta] = meta_field()

    @property
    def value(self) -> str:
        """The time of day value."""
        return f"{self.year}-{self.month}-{self.day}"

    def __str__(self):
        return f"DATE#{self.value}"


@dataclass
@_rule_handler("ldate")
class Ldate(Literal):
    """Long date literal value."""

    year: lark.Token
    month: lark.Token
    day: lark.Token
    meta: Optional[Meta] = meta_field()

    @property
    def value(self) -> str:
        """The long time of day value."""
        return f"{self.year}-{self.month}-{self.day}"

    def __str__(self):
        return f"LDATE#{self.value}"


@dataclass
@_rule_handler("date_and_time")
class DateTime(Literal):
    """Date and time literal value."""

    date: Date
    time: TimeOfDay
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        year: lark.Token,
        month: lark.Token,
        day: lark.Token,
        hour: lark.Token,
        minute: lark.Token,
        second: Optional[lark.Token],
    ) -> DateTime:
        return DateTime(
            date=Date(year=year, month=month, day=day),
            time=TimeOfDay(hour=hour, minute=minute, second=second),
        )

    @property
    def value(self) -> str:
        """The time of day value."""
        return f"{self.date.value}-{self.time.value}"

    def __str__(self):
        return f"DT#{self.value}"


@dataclass
@_rule_handler("ldate_and_time")
class LdateTime(Literal):
    """Long date and time literal value."""

    ldate: Ldate
    ltime: LtimeOfDay
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        year: lark.Token,
        month: lark.Token,
        day: lark.Token,
        hour: lark.Token,
        minute: lark.Token,
        second: Optional[lark.Token],
    ) -> LdateTime:
        return LdateTime(
            ldate=Ldate(year=year, month=month, day=day),
            ltime=LtimeOfDay(hour=hour, minute=minute, second=second),
        )

    @property
    def value(self) -> str:
        """The time of day value."""
        return f"{self.ldate.value}-{self.ltime.value}"

    def __str__(self):
        return f"LDT#{self.value}"


@dataclass
@_rule_handler("string_literal")
class String(Literal):
    """String literal value."""
    value: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return str(self.value)


@as_tagged_union
class Variable(Expression):
    ...


@dataclass
@_rule_handler(
    "indirection_type",
    "pointer_type",
)
class IndirectionType:
    """Indirect access through a pointer or reference."""
    pointer_depth: int
    reference: bool
    meta: Optional[Meta] = meta_field()

    @property
    def is_indirect(self) -> bool:
        """True if this denotes a pointer (of any depth) or a reference."""
        return self.reference or (self.pointer_depth > 0)

    @staticmethod
    def from_lark(*tokens: lark.Token) -> IndirectionType:
        pointer_depth = 0
        reference = False
        upper_tokens = list(map(lambda s: str(s).upper(), tokens))
        if len(tokens) > 0:
            pointer_depth = upper_tokens.count("POINTER TO")
            reference = "REFERENCE TO" in upper_tokens
        return IndirectionType(
            pointer_depth=pointer_depth,
            reference=reference
        )

    @property
    def value(self) -> str:
        return " ".join(
            ["REFERENCE TO"] * self.reference +
            ["POINTER TO"] * self.pointer_depth
        )

    def __str__(self):
        return self.value


@_rule_handler("incomplete_location")
class IncompleteLocation(Enum):
    """Incomplete location information."""
    none = enum.auto()
    input = "%I*"
    output = "%Q*"
    memory = "%M*"

    @staticmethod
    def from_lark(token: Optional[lark.Token]) -> IncompleteLocation:
        return IncompleteLocation(str(token).upper())

    def __str__(self):
        if self == IncompleteLocation.none:
            return ""
        return f"AT {self.value}"


class VariableLocationPrefix(str, Enum):
    input = "I"
    output = "Q"
    memory = "M"

    def __str__(self) -> str:
        return self.value


class VariableSizePrefix(str, Enum):
    bit = "X"
    byte = "B"
    word_16 = "W"
    dword_32 = "D"
    lword_64 = "L"

    def __str__(self) -> str:
        return self.value


@dataclass
@_rule_handler("direct_variable")
class DirectVariable(Variable):
    location_prefix: VariableLocationPrefix
    location: lark.Token
    size_prefix: VariableSizePrefix
    bits: Optional[List[lark.Token]] = None
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        location_prefix: lark.Token,
        size_prefix: Optional[lark.Token],
        location: lark.Token,
        *bits: lark.Token,
    ):
        return DirectVariable(
            location_prefix=VariableLocationPrefix(location_prefix),
            size_prefix=(
                VariableSizePrefix(size_prefix)
                if size_prefix else VariableSizePrefix.bit
            ),
            location=location,
            bits=list(bits) if bits else None,
        )

    def __str__(self) -> str:
        bits = ("." + ".".join(self.bits)) if self.bits else ""
        return f"%{self.location_prefix}{self.size_prefix}{self.location}{bits}"


@_rule_handler("location")
class Location(DirectVariable):
    @staticmethod
    def from_lark(var: DirectVariable) -> Location:
        return Location(
            location_prefix=var.location_prefix,
            location=var.location,
            size_prefix=var.size_prefix,
            bits=var.bits,
            meta=var.meta,
        )

    def __str__(self) -> str:
        direct_loc = super().__str__()
        return f"AT {direct_loc}"


@dataclass
@_rule_handler("variable_name")
class SimpleVariable(Variable):
    name: lark.Token
    dereferenced: bool
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        identifier: lark.Token, dereferenced: Optional[lark.Token]
    ) -> SimpleVariable:
        return SimpleVariable(
            name=identifier,
            dereferenced=dereferenced is not None
        )

    def __str__(self) -> str:
        return f"{self.name}^" if self.dereferenced else f"{self.name}"


@dataclass
@_rule_handler("subscript_list")
class SubscriptList:
    subscripts: List[Expression]
    dereferenced: bool
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args):
        *subscripts, dereferenced = args
        return SubscriptList(
            subscripts=list(subscripts),
            dereferenced=dereferenced is not None,
        )

    def __str__(self) -> str:
        parts = ", ".join(str(subscript) for subscript in self.subscripts)
        return f"[{parts}]^" if self.dereferenced else f"[{parts}]"


@dataclass
@_rule_handler("field_selector")
class FieldSelector:
    field: SimpleVariable
    dereferenced: bool
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(dereferenced: Optional[lark.Token], field: SimpleVariable):
        return FieldSelector(
            field=field,
            dereferenced=dereferenced is not None
        )

    def __str__(self) -> str:
        return f"^.{self.field}" if self.dereferenced else f".{self.field}"


@dataclass
@_rule_handler("multi_element_variable")
class MultiElementVariable(Variable):
    name: SimpleVariable
    dereferenced: bool
    elements: List[Union[SubscriptList, FieldSelector]]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        variable_name: SimpleVariable,
        *subscript_or_field: Union[SubscriptList, FieldSelector]
    ) -> MultiElementVariable:
        return MultiElementVariable(
            name=variable_name,
            elements=list(subscript_or_field),
            dereferenced=False,
        )

    def __str__(self) -> str:
        return "".join(str(part) for part in (self.name, *self.elements))


SymbolicVariable = Union[SimpleVariable, MultiElementVariable]


@dataclass
@_rule_handler("simple_spec_init")
class TypeInitialization:
    indirection: Optional[IndirectionType]
    spec: SimpleSpecification
    value: Optional[Expression]
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> lark.Token:
        """The base type name."""
        return self.spec.type

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        return join_if(self.indirection, " ", self.spec.type)

    def __str__(self) -> str:
        return join_if(self.full_type_name, " := ", self.value)


class Declaration:
    variables: List[DeclaredVariable]
    items: List[Any]
    meta: Optional[Meta]
    init: Union[
        VariableInitDeclaration,
        InputOutputDeclaration,
        OutputDeclaration,
        InputDeclaration,
        GlobalVariableDeclarationType,
    ]


@dataclass
@_rule_handler("simple_type_declaration")
class SimpleTypeDeclaration:
    name: lark.Token
    extends: Optional[Extends]
    init: TypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        if self.extends:
            return f"{self.name} {self.extends} : {self.init}"
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("string_type_declaration")
class StringTypeDeclaration:
    name: lark.Token
    string_type: lark.Token
    length: Optional[lark.Token]
    value: Optional[String]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        type_and_length = join_if(self.string_type, "", self.length)
        type_and_value = join_if(type_and_length, " := ", self.value)
        return f"{self.name} : {type_and_value}"


@dataclass
@_rule_handler("string_type_specification")
class StringTypeSpecification:
    type_name: lark.Token
    # TODO: length includes brackets or parentheses: [255] or (255) or [Const]
    length: Optional[lark.Token] = None
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> lark.Token:
        """The base type name."""
        return self.type_name

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        return join_if(self.type_name, "", self.length)

    def __str__(self) -> str:
        return self.full_type_name


@dataclass
@_rule_handler(
    "single_byte_string_spec",
    "double_byte_string_spec",
)
class StringTypeInitialization:
    spec: StringTypeSpecification
    value: Optional[lark.Token]
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> lark.Token:
        """The base type name."""
        return self.spec.base_type_name

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        return self.spec.full_type_name

    @staticmethod
    def from_lark(
        string_type: lark.Token,
        length: Optional[lark.Token],
        *value_parts: Optional[lark.Token],
    ) -> StringTypeInitialization:
        spec = StringTypeSpecification(string_type, length)
        _, value = value_parts or [None, None]
        return StringTypeInitialization(spec=spec, value=value)

    def __str__(self) -> str:
        return join_if(self.spec, " := ", self.value)


@dataclass
@as_tagged_union
class Subrange:
    ...


@dataclass
class FullSubrange(Subrange):
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return "*"


@dataclass
@_rule_handler("subrange")
class PartialSubrange(Subrange):
    start: Expression
    stop: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.start}..{self.stop}"


@dataclass
@_rule_handler("subrange_specification")
class SubrangeSpecification:
    type_name: lark.Token
    subrange: Optional[Subrange] = None
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> lark.Token:
        """The base type name."""
        return self.type_name

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        if self.subrange:
            return f"{self.type_name} ({self.subrange})"
        return f"{self.type_name}"

    def __str__(self) -> str:
        return self.full_type_name


@dataclass
@_rule_handler("subrange_spec_init")
class SubrangeTypeInitialization:
    indirection: Optional[IndirectionType]
    spec: SubrangeSpecification
    value: Optional[Expression] = None
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> lark.Token:
        """The base type name."""
        return self.spec.base_type_name

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        return self.spec.full_type_name

    def __str__(self) -> str:
        spec = join_if(self.indirection, " ", self.spec)
        if not self.value:
            return spec

        return f"{spec} := {self.value}"


@dataclass
@_rule_handler("subrange_type_declaration")
class SubrangeTypeDeclaration:
    name: lark.Token
    init: SubrangeTypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("enumerated_value")
class EnumeratedValue:
    type_name: Optional[lark.Token]
    name: lark.Token
    value: Optional[Union[Integer, lark.Token]]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        name = join_if(self.type_name, "#", self.name)
        return join_if(name, " := ", self.value)


@dataclass
@_rule_handler("enumerated_specification")
class EnumeratedSpecification:
    _implicit_type_default_: ClassVar[str] = "INT"
    type_name: Optional[lark.Token]
    values: Optional[List[EnumeratedValue]] = None
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> Union[lark.Token, str]:
        """The full type name."""
        return self.type_name or self._implicit_type_default_

    @property
    def full_type_name(self) -> Union[lark.Token, str]:
        """The full type name."""
        return self.base_type_name

    @staticmethod
    def from_lark(*args):
        if len(args) == 1:
            type_name, = args
            return EnumeratedSpecification(type_name=type_name)
        *values, type_name = args
        return EnumeratedSpecification(type_name=type_name, values=list(values))

    def __str__(self) -> str:
        if self.values:
            values = ", ".join(str(value) for value in self.values)
            return join_if(f"({values})", " ", self.type_name)
        return f"{self.type_name}"


@dataclass
@_rule_handler("enumerated_spec_init")
class EnumeratedTypeInitialization:
    indirection: Optional[IndirectionType]
    spec: EnumeratedSpecification
    value: Optional[EnumeratedValue]
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> Union[lark.Token, str]:
        """The base type name."""
        return self.spec.base_type_name

    @property
    def full_type_name(self) -> Union[lark.Token, str]:
        """The full type name."""
        return self.spec.full_type_name

    def __str__(self) -> str:
        spec = join_if(self.indirection, " ", self.spec)
        return join_if(spec, " := ", self.value)


@dataclass
@_rule_handler("enumerated_type_declaration", comments=True)
class EnumeratedTypeDeclaration:
    name: lark.Token
    init: EnumeratedTypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("non_generic_type_name")
class DataType:
    indirection: Optional[IndirectionType]
    type_name: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        if self.indirection and self.indirection.is_indirect:
            return f"{self.indirection} {self.type_name}"
        return f"{self.type_name}"


@dataclass
@_rule_handler("simple_specification")
class SimpleSpecification:
    type: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return str(self.type)


@dataclass
@_rule_handler("indirect_simple_specification")
class IndirectSimpleSpecification:
    indirection: Optional[IndirectionType]
    type: SimpleSpecification
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return join_if(self.indirection, " ", self.type)


@dataclass
@_rule_handler("array_specification")
class ArraySpecification:
    type: Union[DataType, FunctionCall, ObjectInitializerArray]
    subranges: List[Subrange]
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> Union[str, lark.Token]:
        """The base type name."""
        if isinstance(self.type, DataType):
            return self.type.type_name
        return str(self.type.name)

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        return str(self)

    @staticmethod
    def from_lark(*args):
        *subranges, type = args
        if isinstance(type, lark.Token):
            # STRING_TYPE
            type = DataType(
                indirection=None,
                type_name=type,
            )

        return ArraySpecification(type=type, subranges=subranges)

    def __str__(self) -> str:
        subranges = ", ".join(str(subrange) for subrange in self.subranges)
        return f"ARRAY [{subranges}] OF {self.type}"


@dataclass
@_rule_handler("array_initial_element")
class ArrayInitialElement:
    element: ArrayInitialElementType
    count: Optional[Union[EnumeratedValue, Integer]] = None
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        if self.count is None:
            return f"{self.element}"
        return f"{self.count}({self.element})"


@_rule_handler("array_initial_element_count")
class _ArrayInitialElementCount:
    @staticmethod
    def from_lark(
        count: Union[EnumeratedValue, Integer],
        element: ArrayInitialElementType
    ) -> ArrayInitialElement:
        return ArrayInitialElement(
            element=element,
            count=count,
        )


@dataclass
@_rule_handler("bracketed_array_initialization")
class _BracketedArrayInitialization:
    @staticmethod
    def from_lark(*elements: ArrayInitialElement) -> ArrayInitialization:
        return ArrayInitialization(list(elements), brackets=True)


@dataclass
@_rule_handler("bare_array_initialization")
class _BareArrayInitialization:
    @staticmethod
    def from_lark(*elements: ArrayInitialElement) -> ArrayInitialization:
        return ArrayInitialization(list(elements), brackets=False)


@dataclass
class ArrayInitialization:
    elements: List[ArrayInitialElement]
    brackets: bool = False
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        elements = ", ".join(str(element) for element in self.elements)
        if self.brackets:
            return f"[{elements}]"
        return elements


@dataclass
@_rule_handler("object_initializer_array")
class ObjectInitializerArray:
    name: SymbolicVariable
    initializers: List[StructureInitialization]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        function_block_type_name: SymbolicVariable,
        *initializers: List[StructureInitialization]
    ) -> ObjectInitializerArray:
        return ObjectInitializerArray(
            name=function_block_type_name,
            initializers=list(initializers)
        )

    def __str__(self) -> str:
        initializers = ", ".join([f"({init})" for init in self.initializers])
        return f"{self.name}[{initializers}]"


@dataclass
@_rule_handler("array_spec_init")
class ArrayTypeInitialization:
    indirection: Optional[IndirectionType]
    spec: ArraySpecification
    value: Optional[ArrayInitialization]
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> Union[str, lark.Token]:
        """The base type name."""
        return self.spec.base_type_name

    @property
    def full_type_name(self) -> str:
        """The full type name."""
        return self.spec.full_type_name

    def __str__(self) -> str:
        if self.indirection:
            spec = f"{self.indirection} {self.spec}"
        else:
            spec = f"{self.spec}"

        if not self.value:
            return spec

        return f"{spec} := {self.value}"


@dataclass
@_rule_handler("array_type_declaration", comments=True)
class ArrayTypeDeclaration:
    name: lark.Token
    init: ArrayTypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("structure_type_declaration", comments=True)
class StructureTypeDeclaration:
    name: lark.Token
    extends: Optional[Extends]
    indirection: Optional[IndirectionType]
    declarations: List[StructureElementDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        name: lark.Token,
        extends: Optional[Extends],
        indirection: Optional[IndirectionType],
        *declarations: StructureElementDeclaration,
    ):
        return StructureTypeDeclaration(
            name=name,
            extends=extends,
            indirection=indirection,
            declarations=list(declarations),
        )

    def __str__(self) -> str:
        if self.declarations:
            body = "\n".join(
                (
                    "STRUCT",
                    indent("\n".join(str(decl) for decl in self.declarations)),
                    "END_STRUCT",
                )
            )
        else:
            body = "\n".join(("STRUCT", "END_STRUCT"))

        definition = join_if(self.name, " ", self.extends)
        indirection = f" {self.indirection}" if self.indirection else ""
        return f"{definition} :{indirection}\n{body}"


@dataclass
@_rule_handler("structure_element_declaration", comments=True)
class StructureElementDeclaration:
    name: lark.Token
    location: Optional[IncompleteLocation]
    init: Union[
        StructureInitialization,
        ArrayTypeInitialization,
        StringTypeInitialization,
        TypeInitialization,
        SubrangeTypeInitialization,
        EnumeratedTypeInitialization,
        InitializedStructure,
        FunctionCall,
    ]
    meta: Optional[Meta] = meta_field()

    @property
    def variables(self) -> List[str]:
        """API compat"""
        return [self.name]

    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init};"


UnionElementSpecification = Union[
    ArraySpecification,
    SimpleSpecification,
    SubrangeSpecification,
    EnumeratedSpecification,
    IndirectSimpleSpecification,
]


@dataclass
@_rule_handler("union_element_declaration", comments=True)
class UnionElementDeclaration:
    name: lark.Token
    spec: UnionElementSpecification
    meta: Optional[Meta] = meta_field()

    @property
    def variables(self) -> List[str]:
        """API compat"""
        return [self.name]

    def __str__(self) -> str:
        return f"{self.name} : {self.spec};"


@dataclass
@_rule_handler("union_type_declaration", comments=True)
class UnionTypeDeclaration:
    name: lark.Token
    declarations: List[UnionElementDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(name: lark.Token, *decls: UnionElementDeclaration):
        return UnionTypeDeclaration(
            name=name,
            declarations=list(decls),
        )

    def __str__(self) -> str:
        if not self.declarations:
            decls = None
        else:
            decls = indent(
                "\n".join(
                    str(decl) for decl in self.declarations
                )
            )

        return "\n".join(
            line for line in (
                f"{self.name} :",
                "UNION",
                decls,
                "END_UNION",
            )
            if line is not None
        )


@dataclass
@_rule_handler("initialized_structure")
class InitializedStructure:
    name: lark.Token
    init: StructureInitialization
    meta: Optional[Meta] = meta_field()

    @property
    def value(self) -> str:
        """The initialization value (call)."""
        return str(self.init)

    @property
    def base_type_name(self) -> lark.Token:
        """The base type name."""
        return self.name

    @property
    def full_type_name(self) -> lark.Token:
        """The full type name."""
        return self.name

    def __str__(self) -> str:
        return f"{self.name} := {self.init}"


@dataclass
@_rule_handler("structure_initialization")
class StructureInitialization:
    elements: List[StructureElementInitialization]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*elements: StructureElementInitialization):
        return StructureInitialization(elements=list(elements))

    def __str__(self) -> str:
        parts = ", ".join(str(element) for element in self.elements)
        return f"({parts})"


@dataclass
@_rule_handler("structure_element_initialization")
class StructureElementInitialization:
    name: Optional[lark.Token]
    value: Union[
        Constant,
        Expression,
        EnumeratedValue,
        ArrayInitialization,
        StructureInitialization,
    ]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args):
        if len(args) == 1:
            name = None
            value, = args
        else:
            name, value = args
        return StructureElementInitialization(name=name, value=value)

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} := {self.value}"
        return f"{self.value}"


@dataclass
@_rule_handler("unary_expression")
class UnaryOperation(Expression):
    op: lark.Token
    expr: Expression
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(operator: Optional[lark.Token], expr: Expression):
        if operator is None:
            return expr
        return UnaryOperation(
            op=operator,
            expr=expr,
        )

    def __str__(self) -> str:
        if self.op in "-+":
            return f"{self.op}{self.expr}"
        return f"{self.op} {self.expr}"


@dataclass
@_rule_handler(
    "expression",
    "add_expression",
    "and_expression",
    "and_then_expression",
    "or_else_expression",
    "assignment_expression",
    "xor_expression",
    "comparison_expression",
    "equality_expression",
    "power_expression",
    "expression_term"
)
class BinaryOperation(Expression):
    left: Expression
    op: lark.Token
    right: Expression
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(left: Expression, *operator_and_expr: Union[lark.Token, Expression]):
        if not operator_and_expr:
            return left

        def get_operator_and_expr() -> Generator[Tuple[lark.Token, Expression], None, None]:
            operators = typing.cast(Tuple[lark.Token, ...], operator_and_expr[::2])
            expressions = typing.cast(Tuple[Expression, ...], operator_and_expr[1::2])
            yield from zip(operators, expressions)

        binop = None
        for operator, expression in get_operator_and_expr():
            if binop is None:
                binop = BinaryOperation(
                    left=left,
                    op=operator,
                    right=expression
                )
            else:
                binop = BinaryOperation(
                    left=binop,
                    op=operator,
                    right=expression,
                )
        return binop

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"


@dataclass
@_rule_handler("parenthesized_expression")
class ParenthesizedExpression(Expression):
    expr: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"({self.expr})"


@dataclass
@_rule_handler("function_call")
class FunctionCall(Expression):
    name: SymbolicVariable
    parameters: List[ParameterAssignment]
    dereferenced: bool
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> str:
        """
        The base type name.

        This is used as part of the summary mechanism. The "type" is that
        of the underlying function block or function.
        """
        if isinstance(self.name, SimpleVariable):
            return str(self.name.name)
        return str(self.name)

    @property
    def full_type_name(self) -> str:
        """The full type name, including any dereferencing or subscripts."""
        return str(self.name)

    @property
    def value(self) -> str:
        """
        The initialization value (the function call itself).

        This is used as part of the summary tool.
        """
        return str(self)

    @staticmethod
    def from_lark(
        name: SymbolicVariable,
        *params: Union[ParameterAssignment, lark.Token, None],
    ) -> FunctionCall:
        # Condition parameters (which may be `None`) to represent empty tuple
        if params and params[0] is None:
            params = params[1:]
        dereferenced = bool(params and params[-1] == "^")
        if dereferenced:
            params = params[:-1]

        return FunctionCall(
            name=name,
            parameters=typing.cast(List[ParameterAssignment], list(params)),
            dereferenced=dereferenced,
        )

    def __str__(self) -> str:
        dereference = ""
        parameters = ", ".join(str(param) for param in self.parameters)
        if self.dereferenced:
            dereference = "^"
        return f"{self.name}({parameters}){dereference}"


@dataclass
@_rule_handler("var1")
class DeclaredVariable:
    # Alternate name: VariableWithLocation? MaybeLocatedVariable?
    variable: SimpleVariable
    location: Optional[Union[IncompleteLocation, Location]]
    meta: Optional[Meta] = meta_field()

    @property
    def name(self) -> lark.Token:
        """The variable name."""
        return self.variable.name

    @property
    def dereferenced(self) -> bool:
        """Is the variable dereferenced with '^'?."""
        return self.variable.dereferenced

    def __str__(self) -> str:
        return join_if(self.variable, " ", self.location)


@dataclass
class _GenericInit:
    """API compat to give a valid init attribute."""
    # TODO: can we restructure this to be less confusing?
    base_type_name: str
    full_type_name: str
    repr: str
    value: Optional[str]

    def __str__(self) -> str:
        return str(self.repr)


InitDeclarationType = Union[
    TypeInitialization,
    SubrangeTypeInitialization,
    EnumeratedTypeInitialization,
    ArrayTypeInitialization,
    InitializedStructure,
    _GenericInit,  # StringVariableInitDeclaration, EdgeDeclaration
]


class InitDeclaration:
    variables: List[DeclaredVariable]
    init: InitDeclarationType
    meta: Optional[Meta]

    def __str__(self) -> str:
        variables = ", ".join(str(variable) for variable in self.variables)
        return f"{variables} : {self.init}"


@dataclass
@_rule_handler("var1_init_decl", comments=True)
class VariableOneInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    init: Union[TypeInitialization, SubrangeTypeInitialization, EnumeratedTypeInitialization]
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler("array_var_init_decl", comments=True)
class ArrayVariableInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    init: ArrayTypeInitialization
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler("structured_var_init_decl", comments=True)
class StructuredVariableInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    init: InitializedStructure
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler(
    "single_byte_string_var_declaration",
    "double_byte_string_var_declaration",
    comments=True
)
class StringVariableInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    spec: StringTypeSpecification
    value: Optional[lark.Token]
    init: _GenericInit
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(variables: List[DeclaredVariable], string_info: StringTypeInitialization):
        return StringVariableInitDeclaration(
            variables=variables,
            spec=string_info.spec,
            value=string_info.value,
            init=_GenericInit(
                base_type_name=str(string_info.spec.base_type_name),
                full_type_name=str(string_info.spec.full_type_name),
                value=str(string_info.value),
                repr=join_if(string_info.spec, " := ", string_info.value),
            )
        )


@dataclass
@_rule_handler("edge_declaration", comments=True)
class EdgeDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    edge: lark.Token
    meta: Optional[Meta] = meta_field()

    def __post_init__(self):
        full_type_name = f"BOOL {self.edge}"
        self.init = _GenericInit(
            base_type_name="BOOL",
            full_type_name=full_type_name,
            value=None,
            repr=full_type_name,
        )

    def __str__(self):
        variables = ", ".join(str(variable) for variable in self.variables)
        return f"{variables} : {self.init.full_type_name}"


@as_tagged_union
class FunctionBlockDeclaration:
    ...


@dataclass
@_rule_handler("fb_name_decl", comments=True)
class FunctionBlockNameDeclaration(FunctionBlockDeclaration):
    variables: List[lark.Token]   # fb_decl_name_list -> fb_name
    spec: lark.Token
    init: Optional[StructureInitialization] = None
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        variables = ", ".join(self.variables)
        name_and_type = f"{variables} : {self.spec}"
        return join_if(name_and_type, " := ", self.init)


@dataclass
@_rule_handler("fb_invocation_decl", comments=True)
class FunctionBlockInvocationDeclaration(FunctionBlockDeclaration):
    variables: List[lark.Token]
    init: FunctionCall
    defaults: Optional[StructureInitialization] = None
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        variables = ", ".join(self.variables)
        name_and_type = f"{variables} : {self.init}"
        return join_if(name_and_type, " := ", self.defaults)


@as_tagged_union
class ParameterAssignment:
    ...


@dataclass
@_rule_handler("param_assignment")
class InputParameterAssignment(ParameterAssignment):
    name: Optional[SimpleVariable]
    value: Optional[Expression]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args) -> InputParameterAssignment:
        if len(args) == 1:
            value, = args
            name = None
        else:
            name, value = args
        return InputParameterAssignment(name, value)

    def __str__(self) -> str:
        return join_if(self.name, " := ", self.value)


@dataclass
@_rule_handler("output_parameter_assignment")
class OutputParameterAssignment(ParameterAssignment):
    name: SimpleVariable
    value: Optional[Expression]
    inverted: bool = False
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        inverted: Optional[lark.Token],
        name: SimpleVariable,
        value: Expression,
    ) -> OutputParameterAssignment:
        return OutputParameterAssignment(
            name=name, value=value, inverted=inverted is not None
        )

    def __str__(self) -> str:
        prefix = "NOT " if self.inverted else ""
        return prefix + join_if(self.name, " => ", self.value)


AnyLocation = Union[Location, IncompleteLocation]


@dataclass
@_rule_handler("global_var_spec")
class GlobalVariableSpec:
    variables: List[lark.Token]
    location: Optional[AnyLocation]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        name_or_names: Union[lark.Token, lark.Tree],
        location: Optional[AnyLocation] = None
    ) -> GlobalVariableSpec:
        if location is None:
            # Multiple variables without a location
            name_tree = typing.cast(lark.Tree, name_or_names)
            variables = typing.cast(List[lark.Token], name_tree.children)
        else:
            # Only one variable allowed with a location
            variables = typing.cast(List[lark.Token], [name_or_names])
        return GlobalVariableSpec(variables=variables, location=location)

    def __str__(self) -> str:
        if not self.location:
            return ", ".join(self.variables)
        return f"{self.variables[0]} {self.location}"


LocatedVariableSpecInit = Union[
    TypeInitialization,
    SubrangeTypeInitialization,
    EnumeratedTypeInitialization,
    ArrayTypeInitialization,
    InitializedStructure,
    StringTypeInitialization,
]


@dataclass
@_rule_handler("global_var_decl", comments=True)
class GlobalVariableDeclaration:
    spec: GlobalVariableSpec
    init: Union[LocatedVariableSpecInit, FunctionCall]
    meta: Optional[Meta] = meta_field()

    @property
    def variables(self) -> List[lark.Token]:
        """The variable names contained."""
        return self.spec.variables

    @property
    def location(self) -> Optional[AnyLocation]:
        """The (optional) variable location."""
        return self.spec.location

    @property
    def base_type_name(self) -> Union[str, lark.Token]:
        """The base type name of the variable(s)."""
        return self.init.base_type_name

    @property
    def full_type_name(self) -> Union[str, lark.Token]:
        """The full type name of the variable(s)."""
        return self.init.full_type_name

    def __str__(self) -> str:
        return f"{self.spec} : {self.init}"


@dataclass
@_rule_handler("extends")
class Extends:
    name: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"EXTENDS {self.name}"


@dataclass
@_rule_handler("implements")
class Implements:
    interfaces: List[lark.Token]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        *interfaces: lark.Token,
    ) -> Implements:
        return Implements(interfaces=list(interfaces))

    def __str__(self) -> str:
        return "IMPLEMENTS " + ", ".join(self.interfaces)


@dataclass
@_rule_handler("function_block_type_declaration", comments=True)
class FunctionBlock:
    name: lark.Token
    access: Optional[AccessSpecifier]
    extends: Optional[Extends]
    implements: Optional[Implements]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBlockBody]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        fb_token: lark.Token,
        access: Optional[AccessSpecifier],
        derived_name: lark.Token,
        extends: Optional[Extends],
        implements: Optional[Implements],
        *args
    ) -> FunctionBlock:
        *declarations, body, _ = args
        return FunctionBlock(
            name=derived_name,
            access=access,
            extends=extends,
            implements=implements,
            declarations=list(declarations),
            body=body,
        )

    def __str__(self) -> str:
        access_and_name = join_if(self.access, " ", self.name)
        header = f"FUNCTION_BLOCK {access_and_name}"
        header = join_if(header, " ", self.implements)
        header = join_if(header, " ", self.extends)
        return "\n".join(
            line for line in
            (
                header,
                *[str(declaration) for declaration in self.declarations],
                indent_if(self.body),
                "END_FUNCTION_BLOCK",
            )
            if line is not None
        )


@dataclass
@_rule_handler("function_declaration", comments=True)
class Function:
    access: Optional[AccessSpecifier]
    name: lark.Token
    return_type: Optional[Union[SimpleSpecification, IndirectSimpleSpecification]]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBody]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        access: Optional[AccessSpecifier],
        name: lark.Token,
        return_type: Optional[Union[SimpleSpecification, IndirectSimpleSpecification]],
        *remainder
    ) -> Function:
        *declarations, body = remainder
        return Function(
            name=name,
            access=access,
            return_type=return_type,
            declarations=typing.cast(
                List[VariableDeclarationBlock], list(declarations)
            ),
            body=typing.cast(Optional[FunctionBody], body),
        )

    def __str__(self) -> str:
        access_and_name = join_if(self.access, " ", self.name)
        function = f"FUNCTION {access_and_name}"
        return_type = f": {self.return_type}" if self.return_type else None
        return "\n".join(
            line for line in
            (
                join_if(function, " ", return_type),
                *[indent_if(declaration) for declaration in self.declarations],
                indent_if(self.body),
                "END_FUNCTION",
            )
            if line is not None
        )


@dataclass
@_rule_handler("program_declaration", comments=True)
class Program:
    name: lark.Token
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBody]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return "\n".join(
            s for s in (
                f"PROGRAM {self.name}",
                *[indent_if(decl) for decl in self.declarations],
                indent_if(self.body),
                "END_PROGRAM",
            )
            if s is not None
        )


@dataclass
@_rule_handler("action", comments=True)
class Action:
    name: lark.Token
    body: Optional[FunctionBody]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return "\n".join(
            line for line in
            (
                f"ACTION {self.name}:",
                indent_if(self.body),
                "END_ACTION",
            )
            if line is not None
        )


@dataclass
@_rule_handler("function_block_method_declaration", comments=True)
class Method:
    access: Optional[AccessSpecifier]
    name: lark.Token
    return_type: Optional[LocatedVariableSpecInit]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBody]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        access: Optional[AccessSpecifier],
        name: lark.Token,
        return_type: Optional[LocatedVariableSpecInit],
        *args
    ) -> Method:
        *declarations, body = args
        return Method(
            name=name,
            access=access,
            return_type=return_type,
            declarations=list(declarations),
            body=body,
        )

    def __str__(self) -> str:
        access_and_name = join_if(self.access, " ", self.name)
        method = join_if(access_and_name, " : ", self.return_type)
        return "\n".join(
            line for line in
            (
                f"METHOD {method}",
                *[indent_if(declaration) for declaration in self.declarations],
                indent_if(self.body),
                "END_METHOD",
            )
            if line is not None
        )


@dataclass
@_rule_handler("function_block_property_declaration", comments=True)
class Property:
    access: Optional[AccessSpecifier]
    name: lark.Token
    return_type: Optional[LocatedVariableSpecInit]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBody]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        access: Optional[AccessSpecifier],
        name: lark.Token,
        return_type: Optional[LocatedVariableSpecInit],
        *args
    ) -> Property:
        *declarations, body = args
        return Property(
            name=name,
            access=access,
            return_type=return_type,
            declarations=list(declarations),
            body=body,
        )

    def __str__(self) -> str:
        access_and_name = join_if(self.access, " ", self.name)
        property = join_if(access_and_name, " : ", self.return_type)
        return "\n".join(
            line for line in
            (
                f"PROPERTY {property}",
                *[indent_if(declaration) for declaration in self.declarations],
                indent_if(self.body),
                "END_PROPERTY",
            )
            if line is not None
        )


VariableInitDeclaration = Union[
    ArrayVariableInitDeclaration,
    StringVariableInitDeclaration,
    VariableOneInitDeclaration,
    FunctionBlockDeclaration,
    EdgeDeclaration,
    StructuredVariableInitDeclaration,
]

InputOutputDeclaration = VariableInitDeclaration
OutputDeclaration = VariableInitDeclaration

InputDeclaration = Union[
    VariableInitDeclaration,
    EdgeDeclaration,
]
GlobalVariableDeclarationType = Union[
    VariableInitDeclaration,
    GlobalVariableDeclaration,
]


@as_tagged_union
class VariableDeclarationBlock:
    block_header: ClassVar[str] = "VAR"
    items: List[Any]
    meta: Optional[Meta]

    @property
    def attribute_pragmas(self) -> List[str]:
        """Attribute pragmas associated with the variable declaration block."""
        # TODO: deprecate
        return getattr(self.meta, "attribute_pragmas", [])


@dataclass
@_rule_handler("var_declarations", comments=True)
class VariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR"
    attrs: Optional[VariableAttributes]
    items: List[VariableInitDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes], tree: lark.Tree
    ) -> VariableDeclarations:
        items = typing.cast(List[VariableInitDeclaration], tree.children)
        return VariableDeclarations(attrs=attrs, items=items)

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("static_var_declarations", comments=True)
class StaticDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_STAT"
    attrs: Optional[VariableAttributes]
    items: List[VariableInitDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes],
        tree: lark.Tree,
    ) -> StaticDeclarations:
        items = typing.cast(List[VariableInitDeclaration], tree.children)
        return StaticDeclarations(attrs, list(items))

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_STAT", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("temp_var_decls", comments=True)
class TemporaryVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_TEMP"
    items: List[VariableInitDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(items: lark.Tree) -> TemporaryVariableDeclarations:
        return TemporaryVariableDeclarations(
            typing.cast(List[VariableInitDeclaration], items.children)
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_TEMP",
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("var_inst_declaration", comments=True)
class MethodInstanceVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_INST"
    attrs: Optional[VariableAttributes]
    items: List[VariableInitDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes], items: lark.Tree
    ) -> MethodInstanceVariableDeclarations:
        return MethodInstanceVariableDeclarations(
            attrs=attrs,
            items=typing.cast(
                List[VariableInitDeclaration],
                items.children
            )
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_INST",
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("located_var_decl", comments=True)
class LocatedVariableDeclaration:
    name: Optional[SimpleVariable]
    location: Location
    init: LocatedVariableSpecInit
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init}"


@dataclass
@_rule_handler("located_var_declarations", comments=True)
class LocatedVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR"
    attrs: Optional[VariableAttributes]
    items: List[LocatedVariableDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes],
        *items: LocatedVariableDeclaration,
    ) -> LocatedVariableDeclarations:
        return LocatedVariableDeclarations(
            attrs=attrs,
            items=list(items),
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


IncompleteLocatedVariableSpecInit = Union[
    SimpleSpecification,
    TypeInitialization,
    SubrangeTypeInitialization,
    EnumeratedTypeInitialization,
    ArrayTypeInitialization,
    InitializedStructure,
    StringTypeSpecification,
]


@dataclass
@_rule_handler("incomplete_located_var_decl", comments=True)
class IncompleteLocatedVariableDeclaration:
    name: SimpleVariable
    location: IncompleteLocation
    init: IncompleteLocatedVariableSpecInit
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init}"


@dataclass
@_rule_handler("incomplete_located_var_declarations", comments=True)
class IncompleteLocatedVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR"
    attrs: Optional[VariableAttributes]
    items: List[IncompleteLocatedVariableDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes],
        *items: IncompleteLocatedVariableDeclaration,
    ) -> IncompleteLocatedVariableDeclarations:
        return IncompleteLocatedVariableDeclarations(
            attrs=attrs,
            items=list(items),
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("external_declaration", comments=True)
class ExternalVariableDeclaration:
    name: lark.Token
    spec: Union[
        SimpleSpecification,
        lark.Token,  # SIMPLE_SPECIFICATION / STRUCTURE_TYPE_NAME / FUNCTION_BLOCK_TYPE_NAME
        SubrangeSpecification,
        EnumeratedSpecification,
        ArraySpecification,
    ]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.spec}"


@dataclass
@_rule_handler("external_var_declarations", comments=True)
class ExternalVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_EXTERNAL"
    attrs: Optional[VariableAttributes]
    items: List[ExternalVariableDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes],
        *items: ExternalVariableDeclaration,
    ) -> ExternalVariableDeclarations:
        return ExternalVariableDeclarations(
            attrs=attrs,
            items=list(items),
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_EXTERNAL", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("input_declarations", comments=True)
class InputDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_INPUT"
    attrs: Optional[VariableAttributes]
    items: List[InputDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes], *items: InputDeclaration
    ) -> InputDeclarations:
        return InputDeclarations(attrs, list(items) if items else [])

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_INPUT", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("output_declarations", comments=True)
class OutputDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_OUTPUT"
    attrs: Optional[VariableAttributes]
    items: List[OutputDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes], items: lark.Tree
    ) -> OutputDeclarations:
        return OutputDeclarations(
            attrs, typing.cast(List[OutputDeclaration], items.children)
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_OUTPUT", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("input_output_declarations", comments=True)
class InputOutputDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_IN_OUT"
    attrs: Optional[VariableAttributes]
    items: List[InputOutputDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes], items: lark.Tree
    ) -> InputOutputDeclarations:
        return InputOutputDeclarations(
            attrs,
            typing.cast(List[InputOutputDeclaration], items.children)
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_IN_OUT", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("program_access_decl", comments=True)
class AccessDeclaration:
    name: lark.Token
    variable: SymbolicVariable
    type: DataType
    direction: Optional[lark.Token]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return join_if(
            f"{self.name} : {self.variable} : {self.type}",
            " ",
            self.direction
        )


@dataclass
@_rule_handler("function_var_declarations", comments=True)
class FunctionVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR"
    attrs: Optional[VariableAttributes]
    items: List[VariableInitDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        attrs: Optional[VariableAttributes],
        body: lark.Tree,
    ) -> FunctionVariableDeclarations:
        items = typing.cast(List[VariableInitDeclaration], body.children)
        return FunctionVariableDeclarations(
            attrs=attrs,
            items=items,
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("program_access_decls", comments=True)
class AccessDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_ACCESS"
    items: List[AccessDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*items: AccessDeclaration) -> AccessDeclarations:
        return AccessDeclarations(list(items))

    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_ACCESS",
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("global_var_declarations", comments=True)
class GlobalVariableDeclarations(VariableDeclarationBlock):
    block_header: ClassVar[str] = "VAR_GLOBAL"
    attrs: Optional[GlobalVariableAttributes]
    items: List[GlobalVariableDeclaration]
    meta: Optional[Meta] = meta_field()
    name: Optional[str] = None

    @staticmethod
    def from_lark(
        attrs: Optional[GlobalVariableAttributes],
        *items: GlobalVariableDeclaration
    ) -> GlobalVariableDeclarations:
        return GlobalVariableDeclarations(
            name=None,  # This isn't in the code; set later
            attrs=attrs,
            items=list(items)
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_GLOBAL", " ", self.attrs),
                *(indent(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@as_tagged_union
class Statement:
    ...


@_rule_handler("function_call_statement", comments=True)
class FunctionCallStatement(Statement, FunctionCall):
    @staticmethod
    def from_lark(
        invocation: FunctionCall,
    ) -> FunctionCallStatement:
        return FunctionCallStatement(
            name=invocation.name,
            parameters=invocation.parameters,
            dereferenced=invocation.dereferenced,
            meta=invocation.meta,
        )

    def __str__(self):
        invoc = super().__str__()
        return f"{invoc};"


@dataclass
@_rule_handler("chained_function_call_statement", comments=True)
class ChainedFunctionCallStatement(Statement):
    invocations: List[FunctionCall]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*invocations: FunctionCall) -> ChainedFunctionCallStatement:
        return ChainedFunctionCallStatement(
            invocations=list(invocations)
        )

    def __str__(self) -> str:
        invoc = ".".join(str(invocation) for invocation in self.invocations)
        return f"{invoc};"


@dataclass
@_rule_handler("else_if_clause", comments=True)
class ElseIfClause:
    if_expression: Expression
    statements: Optional[StatementList]
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "\n".join(
            s for s in (
                f"ELSIF {self.if_expression} THEN",
                indent_if(self.statements),
            )
            if s is not None
        )


@dataclass
@_rule_handler("else_clause", comments=True)
class ElseClause:
    statements: Optional[StatementList]
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "\n".join(
            s for s in (
                "ELSE",
                indent_if(self.statements),
            )
            if s is not None
        )


@dataclass
@_rule_handler("if_statement", comments=True)
class IfStatement(Statement):
    if_expression: Expression
    statements: Optional[StatementList]
    else_ifs: List[ElseIfClause]
    else_clause: Optional[ElseClause]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        if_expr: Expression,
        then: Optional[StatementList],
        *args: Optional[Union[ElseIfClause, ElseClause]]
    ) -> IfStatement:
        else_clause: Optional[ElseClause] = None
        if args and isinstance(args[-1], ElseClause) or args[-1] is None:
            else_clause = args[-1]
            args = args[:-1]

        else_ifs = typing.cast(List[ElseIfClause], list(args))
        return IfStatement(
            if_expression=if_expr,
            statements=then,
            else_ifs=else_ifs,
            else_clause=else_clause,
        )

    def __str__(self):
        return "\n".join(
            s for s in (
                f"IF {self.if_expression} THEN",
                indent_if(self.statements),
                *[str(else_if) for else_if in self.else_ifs],
                str(self.else_clause) if self.else_clause else None,
                "END_IF",
            )
            if s is not None
        )


CaseMatch = Union[
    Subrange,
    Integer,
    EnumeratedValue,
    SymbolicVariable,
    BitString,
    Boolean,
]


@dataclass
@_rule_handler("case_element", comments=True)
class CaseElement(Statement):
    matches: List[CaseMatch]
    statements: Optional[StatementList]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        matches: lark.Tree,
        statements: Optional[StatementList],
    ) -> CaseElement:
        return CaseElement(
            matches=typing.cast(List[CaseMatch], matches.children),
            statements=statements,
        )

    def __str__(self):
        matches = ", ".join(str(match) for match in self.matches)
        return "\n".join(
            s for s in (
                f"{matches}:",
                indent_if(self.statements),
            )
            if s is not None
        )


@dataclass
@_rule_handler("case_statement", comments=True)
class CaseStatement(Statement):
    expression: Expression
    cases: List[CaseElement]
    else_clause: Optional[ElseClause]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return "\n".join(
            s for s in (
                f"CASE {self.expression} OF",
                *[str(case) for case in self.cases],
                str(self.else_clause) if self.else_clause else None,
                "END_CASE",
            )
            if s is not None
        )


@dataclass
@_rule_handler("no_op_statement", comments=True)
class NoOpStatement(Statement):
    variable: Variable
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable};"


@dataclass
@_rule_handler("action_statement", comments=True)
class ActionStatement(Statement):
    # TODO: overlaps with no-op statement?
    action: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.action};"


@dataclass
@_rule_handler("set_statement", comments=True)
class SetStatement(Statement):
    variable: SymbolicVariable
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable} S= {self.expression};"


@dataclass
@_rule_handler("reference_assignment_statement", comments=True)
class ReferenceAssignmentStatement(Statement):
    variable: SymbolicVariable
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable} REF= {self.expression};"


@dataclass
@_rule_handler("reset_statement")
class ResetStatement(Statement):
    variable: SymbolicVariable
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable} R= {self.expression};"


@dataclass
@_rule_handler("exit_statement", comments=True)
class ExitStatement(Statement):
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "EXIT;"


@dataclass
@_rule_handler("continue_statement", comments=True)
class ContinueStatement(Statement):
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "CONTINUE;"


@dataclass
@_rule_handler("return_statement", comments=True)
class ReturnStatement(Statement):
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "RETURN;"


@dataclass
@_rule_handler("assignment_statement", comments=True)
class AssignmentStatement(Statement):
    variables: List[Variable]
    expression: Expression
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args) -> AssignmentStatement:
        *variables, expression = args
        return AssignmentStatement(
            variables=list(variables),
            expression=expression
        )

    def __str__(self):
        variables = " := ".join(str(var) for var in self.variables)
        return f"{variables} := {self.expression};"


@dataclass
@_rule_handler("while_statement", comments=True)
class WhileStatement(Statement):
    expression: Expression
    statements: StatementList
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "\n".join(
            s for s in (
                f"WHILE {self.expression}",
                "DO",
                indent_if(self.statements),
                "END_WHILE",
            )
            if s is not None
        )


@dataclass
@_rule_handler("repeat_statement", comments=True)
class RepeatStatement(Statement):
    statements: StatementList
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "\n".join(
            s for s in (
                "REPEAT",
                indent_if(self.statements),
                f"UNTIL {self.expression}",
                "END_REPEAT",
            )
            if s is not None
        )


@dataclass
@_rule_handler("for_statement", comments=True)
class ForStatement(Statement):
    control: SymbolicVariable
    from_: Expression
    to: Expression
    step: Optional[Expression]
    statements: StatementList
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        step = f" BY {self.step}" if self.step else ""
        return "\n".join(
            line for line in (
                f"FOR {self.control} := {self.from_} TO {self.to}{step}",
                "DO",
                indent_if(self.statements),
                "END_FOR",
            )
            if line is not None
        )


@dataclass
@_rule_handler("statement_list")
class StatementList:
    statements: List[Statement]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*statements: Statement) -> StatementList:
        return StatementList(
            statements=list(statements)
        )

    def __str__(self) -> str:
        return "\n".join(str(statement) for statement in self.statements)


FunctionBlockBody = Union[
    StatementList,
]

FunctionBody = FunctionBlockBody  # Identical, currently


TypeDeclarationItem = Union[
    ArrayTypeDeclaration,
    StructureTypeDeclaration,
    StringTypeDeclaration,
    SimpleTypeDeclaration,
    SubrangeTypeDeclaration,
    EnumeratedTypeDeclaration,
    UnionTypeDeclaration,
]


@dataclass
@_rule_handler("data_type_declaration", comments=True)
class DataTypeDeclaration:
    declaration: Optional[TypeDeclarationItem]
    access: Optional[AccessSpecifier]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        access: Optional[AccessSpecifier],
        declaration: Optional[TypeDeclarationItem] = None,
    ) -> DataTypeDeclaration:
        return DataTypeDeclaration(access=access, declaration=declaration)

    def __str__(self) -> str:
        if not self.declaration:
            return "TYPE\nEND_TYPE"

        decl = indent(self.declaration).lstrip()
        if not isinstance(
            self.declaration, (StructureTypeDeclaration, UnionTypeDeclaration)
        ):
            # note: END_STRUCT; END_UNION; result in "END_TYPE expected not ;"
            decl = decl + ";"

        decl = join_if(self.access, " ", decl)

        return "\n".join(
            (
                f"TYPE {decl}",
                "END_TYPE",
            )
        )


SourceCodeItem = Union[
    DataTypeDeclaration,
    Function,
    FunctionBlock,
    Action,
    Method,
    Program,
    GlobalVariableDeclarations,
]


@dataclass
@_rule_handler("iec_source")
class SourceCode:
    """Top-level source code item."""
    items: List[SourceCodeItem]
    filename: Optional[pathlib.Path] = None
    raw_source: Optional[str] = None
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args: SourceCodeItem) -> SourceCode:
        return SourceCode(list(args))

    def __str__(self):
        return "\n".join(str(item) for item in self.items)


def _annotator_wrapper(handler):
    def wrapped(self: GrammarTransformer, data: Any, children: list, meta: lark.tree.Meta) -> Any:
        result = handler(*children)
        if result is not None and not isinstance(result, (lark.Tree, lark.Token, list)):
            result.meta = Meta.from_lark(meta)
        return result

    return wrapped


def _annotator_method_wrapper(handler):
    def wrapped(self: GrammarTransformer, data: Any, children: list, meta: lark.tree.Meta) -> Any:
        result = handler(self, *children)
        if result is not None and not isinstance(result, (lark.Tree, lark.Token, list)):
            result.meta = Meta.from_lark(meta)
        return result

    return wrapped


class GrammarTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`SourceCode`.

    Attributes
    ----------
    _filename : str
        Filename of grammar being transformed.

    comments : list of lark.Token
        Sorted list of comments and pragmas for annotating the resulting
        transformed grammar.
    """
    _filename: Optional[pathlib.Path]
    comments: List[lark.Token]

    def __init__(
        self,
        comments: Optional[List[lark.Token]] = None,
        fn: Optional[AnyPath] = None,
        source_code: Optional[str] = None,
    ):
        super().__init__()
        self._filename = pathlib.Path(fn) if fn else None
        self._source_code = source_code
        self.comments = comments or []

    locals().update(
        **dict(
            (str(name), _annotator_wrapper(handler))
            for name, handler in _class_handlers.items()
        )
    )

    def transform(self, tree):
        transformed = super().transform(tree)
        if self.comments:
            merge_comments(transformed, self.comments)
        if isinstance(transformed, SourceCode):
            transformed.raw_source = self._source_code
            transformed.filename = (
                self._filename
                if self._filename is not None else None
            )
            for item in transformed.items:
                if isinstance(item, GlobalVariableDeclarations):
                    item.name = self._filename.stem if self._filename else None

        return transformed

    @_annotator_method_wrapper
    def constant(self, constant: Constant) -> Constant:
        return constant

    @_annotator_method_wrapper
    def full_subrange(self):
        return FullSubrange()

    @_annotator_method_wrapper
    def var1_list(self, *items: DeclaredVariable) -> List[DeclaredVariable]:
        return list(items)

    @_annotator_method_wrapper
    def fb_decl_name_list(self, *items: lark.Token) -> List[lark.Token]:
        return list(items)

    @_annotator_method_wrapper
    def signed_integer(self, value: lark.Token):
        return Integer.from_lark(None, value)

    @_annotator_method_wrapper
    def integer(self, value: lark.Token):
        return Integer.from_lark(None, value)

    @_annotator_method_wrapper
    def binary_integer(self, value: Union[Integer, lark.Token]):
        return Integer.from_lark(None, value, base=2)

    @_annotator_method_wrapper
    def octal_integer(self, value: Union[Integer, lark.Token]):
        return Integer.from_lark(None, value, base=8)

    @_annotator_method_wrapper
    def hex_integer(self, value: Union[Integer, lark.Token]):
        return Integer.from_lark(None, value, base=16)

    @_annotator_method_wrapper
    def true(self, value: lark.Token):
        return Boolean(value=value)

    @_annotator_method_wrapper
    def false(self, value: lark.Token):
        return Boolean(value=value)

    @_annotator_method_wrapper
    def program_var_declarations(self, *declarations: VariableDeclarationBlock):
        return list(declarations)

    @_annotator_method_wrapper
    def case_elements(self, *cases: CaseStatement):
        return list(cases)

    def __default__(self, data, children, meta):
        """
        Default function that is called if there is no attribute matching ``data``
        """
        return lark.Tree(data, children, meta)

    def _call_userfunc(self, tree, new_children=None):
        """
        Assumes tree is already transformed

        Re-implementation of lark.visitors.Transformer to make the code paths
        easier to follow.  May break based on upstream API.
        """
        children = new_children if new_children is not None else tree.children
        try:
            handler = getattr(self, tree.data)
        except AttributeError:
            return self.__default__(tree.data, children, tree.meta)

        return handler(tree.data, children, tree.meta)


def merge_comments(source: Any, comments: List[lark.Token]):
    """
    Take the transformed tree and annotate comments back into meta information.
    """
    if source is None or not comments:
        return

    if isinstance(source, (lark.Tree, lark.Token)):
        ...
    elif isinstance(source, (list, tuple)):
        for item in source:
            merge_comments(item, comments)
    elif is_dataclass(source):
        meta = getattr(source, "meta", None)
        if meta:
            if type(source) in _comment_consumers:
                if not hasattr(meta, "comments"):
                    meta.comments = []
                while comments and comments[0].line <= meta.line:
                    meta.comments.append(comments.pop(0))
        for field in fields(source):
            obj = getattr(source, field.name, None)
            if obj is not None:
                merge_comments(obj, comments)


Constant = Union[
    Duration,
    Lduration,
    TimeOfDay,
    Date,
    DateTime,
    Ldate,
    LdateTime,
    Real,
    Integer,
    String,
    BitString,
    Boolean,
]


ArrayInitialElementType = Union[
    Constant,
    StructureInitialization,
    EnumeratedValue,
]


if apischema is not None:
    # Optional apischema deserializers

    @apischema.deserializer
    def _method_access_deserializer(access: int) -> AccessSpecifier:
        return AccessSpecifier(access)

    @apischema.deserializer
    def _var_attrs_deserializer(attrs: int) -> VariableAttributes:
        return VariableAttributes(attrs)

    @apischema.deserializer
    def _global_var_attrs_deserializer(attrs: int) -> GlobalVariableAttributes:
        return GlobalVariableAttributes(attrs)
