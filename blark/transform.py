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

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import lark

from . import util
from .util import AnyPath, maybe_add_brackets, rebuild_lark_tree_with_line_map

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


@dataclasses.dataclass
class FormatSettings:
    indent: str = "    "


_format_settings = FormatSettings()


def configure_formatting(settings: FormatSettings):
    """Override the default code formatting settings."""
    global _format_settings
    _format_settings = settings


def multiline_code_block(block: str) -> str:
    """Multiline code block with lax beginning/end newlines."""
    return textwrap.dedent(block.strip("\n")).rstrip()


def join_if(value1: Optional[Any], delimiter: str, value2: Optional[Any]) -> str:
    """'{value1}{delimiter}{value2} if value1 and value2, otherwise just {value1} or {value2}."""
    return delimiter.join(
        str(value) for value in (value1, value2)
        if value is not None
    )


def indent_if(value: Optional[Any], prefix: Optional[str] = None) -> Optional[str]:
    """Stringified and indented {value} if not None."""
    if value is not None:
        if prefix is None:
            prefix = _format_settings.indent
        return textwrap.indent(str(value), prefix)
    return None


def indent(value: Any, prefix: Optional[str] = None) -> str:
    """Stringified and indented {value}."""
    if prefix is None:
        prefix = _format_settings.indent
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
    """Lark-derived meta information in the form of a dataclass."""
    #: If the metadata information is not yet filled.
    empty: bool = True
    #: Column number.
    column: Optional[int] = None
    #: Comments relating to the line.
    comments: List[lark.Token] = dataclasses.field(default_factory=list)
    #: Containing start column.
    container_column: Optional[int] = None
    #: Containing end column.
    container_end_column: Optional[int] = None
    #: Containing end line.
    container_end_line: Optional[int] = None
    #: Containing start line.
    container_line: Optional[int] = None
    #: Final column number.
    end_column: Optional[int] = None
    #: Final line number.
    end_line: Optional[int] = None
    #: Final character position.
    end_pos: Optional[int] = None
    #: Line number.
    line: Optional[int] = None
    #: Starting character position.
    start_pos: Optional[int] = None

    @staticmethod
    def from_lark(lark_meta: lark.tree.Meta) -> Meta:
        """Generate a Meta instance from the lark Metadata."""
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


def get_grammar_for_class(cls: type) -> Dict[str, str]:
    """
    Given a class, get blark's ``iec.lark`` associated grammar definition(s).
    """
    matches = {}
    for rule, othercls in _rule_to_class.items():
        if othercls is cls:
            matches[rule] = "unknown"

    if not matches:
        return matches

    for rule in list(matches):
        matches[rule] = util.get_grammar_for_rule(rule)

    return matches


class _FlagHelper:
    """A helper base class which translates tokens to ``enum.Flag`` instances."""

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


@dataclasses.dataclass(frozen=True)
class TypeInformation:
    """Type information derived from a specification or initialization."""

    base_type_name: Union[str, lark.Token]
    full_type_name: Union[str, lark.Token]
    context: Any

    @classmethod
    def from_init(
        cls: Type[Self],
        init: Union[
            StructureInitialization,
            ArrayTypeInitialization,
            StringTypeInitialization,
            TypeInitialization,
            SubrangeTypeInitialization,
            EnumeratedTypeInitialization,
            InitializedStructure,
            FunctionCall,
        ],
    ) -> Self:
        if isinstance(init, StructureInitialization):
            return UnresolvedTypeInformation(  # TODO
                base_type_name="",
                full_type_name="",
                context=init,
            )
        if isinstance(init, InitializedStructure):
            return cls(
                base_type_name=init.name,
                full_type_name=init.name,
                context=init,
            )
        if isinstance(init, FunctionCall):
            # May be multi-element variable referenve; stringified here.
            return cls(
                base_type_name=str(init.name),
                full_type_name=str(init.name),
                context=init,
            )
        spec_type = cls.from_spec(init.spec)
        if isinstance(init, TypeInitialization):
            return cls(
                base_type_name=spec_type.base_type_name,
                full_type_name=spec_type.full_type_name,
                context=init,
            )
        return spec_type

    @classmethod
    def from_spec(
        cls: Type[Self],
        spec: Union[
            ArraySpecification,
            DataType,
            EnumeratedSpecification,
            FunctionCall,
            IndirectSimpleSpecification,
            ObjectInitializerArray,
            SimpleSpecification,
            StringTypeSpecification,
            SubrangeSpecification,
        ],
    ) -> Self:
        full_type_name = str(spec)
        if isinstance(spec, DataType):
            if isinstance(spec.type_name, StringTypeSpecification):
                base_type_name = str(spec)
            else:
                base_type_name = spec.type_name
        elif isinstance(spec, ArraySpecification):
            base_type_name = spec.base_type_name
        elif isinstance(spec, StringTypeSpecification):
            base_type_name = spec.base_type_name
        elif isinstance(spec, EnumeratedSpecification):
            base_type_name = str(spec.type_name or spec._implicit_type_default_)
            full_type_name = base_type_name
        elif isinstance(spec, (SimpleSpecification, IndirectSimpleSpecification)):
            base_type_name = str(spec.type)
        elif isinstance(spec, SubrangeSpecification):
            base_type_name = str(spec.type_name)
        elif isinstance(spec, FunctionCall):
            base_type_name = spec.base_type_name
        else:
            # base_type_name = str(spec.name)
            raise NotImplementedError(spec)
        return cls(
            base_type_name=base_type_name,
            full_type_name=full_type_name,
            context=spec,
        )


@dataclasses.dataclass(frozen=True)
class UnresolvedTypeInformation(TypeInformation):
    ...


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
    """
    Base class for all types of expressions.

    This includes all literals (integers, etc.) and more complicated
    mathematical expressions.

    Marked as a "tagged union" so that serialization will uniquely identify the
    Python class.
    """

    def __str__(self) -> str:
        raise NotImplementedError


@as_tagged_union
class Literal(Expression):
    """
    Base class for all literal values.

    Marked as a "tagged union" so that serialization will uniquely identify the
    Python class.
    """


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

    #: The optional type name of the string.
    type_name: Optional[lark.Token]
    #: The string literal.
    value: lark.Token
    #: The numeric base of the value (e.g., 10 is decimal)
    base: ClassVar[int] = 10
    #: Lark metadata.
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
    """
    Variable base class.

    Marked as a "tagged union" so that serialization will uniquely identify the
    Python class.

    Includes:

    1. Direct variables with I/O linkage (e.g., ``var AT %I*``); may be
       located (e.g., ``AT %IX1.1``) or incomplete (e.g., just ``%I*``).
    2. "Simple", single-element variables (referenced by name, potentially
        dereferenced pointers) (e.g., ``var`` or ``var^``).
    3. Multi-element variables (e.g., ``a.b.c`` or ``a^.b[1].c``).
    """
    ...


@dataclass
@_rule_handler(
    "indirection_type",
    "pointer_type",
)
class IndirectionType:
    """Indirect access through a pointer or reference."""
    #: A depth of 1 is "POINTER TO", a depth of 2 is "POINTER TO POINTER TO".
    pointer_depth: int
    #: If set, "REFERENCE TO POINTER TO..."
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
    #: I/O to PLC task.
    input = "%I*"
    #: PLC task to I/O.
    output = "%Q*"
    #: Memory.
    memory = "%M*"

    @staticmethod
    def from_lark(token: Optional[lark.Token]) -> IncompleteLocation:
        return IncompleteLocation(str(token).upper())

    def __str__(self):
        if self == IncompleteLocation.none:
            return ""
        return f"AT {self.value}"


class VariableLocationPrefix(str, Enum):
    #: I/O to PLC task.
    input = "I"
    #: PLC task to I/O.
    output = "Q"
    memory = "M"

    def __str__(self) -> str:
        return self.value


class VariableSizePrefix(str, Enum):
    """Size prefix, used in locations (e.g., ``%IX1.1`` has a bit prefix)."""
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
    """
    Direct variables with I/O linkage.

    Example: ``var AT %I*``

    May be located (e.g., ``AT %IX1.1``) or incomplete (e.g., just ``%I*``).
    """

    #: The location prefix (e.g., I, Q, or M)
    location_prefix: VariableLocationPrefix
    #: The location number itself (e.g., 2 of %IX2.1)
    location: lark.Token
    #: Size prefix, used in locations (e.g., ``%IX1.1`` has a bit prefix).
    size_prefix: VariableSizePrefix
    #: The number of bits.
    bits: Optional[List[lark.Token]] = None
    #: Lark metadata.
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
    """A located direct variable. (e.g., ``AT %IX1.1``)"""

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
    """
    A simple, single-element variable.

    Specified by name, may potentially be dereferenced pointers.
    Examples::

        var
        var^
    """
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
    """
    A list of subscripts.

    Examples::

        [1, 2, 3]
        [Constant, GVL.Value, 1 + 3]
        [1]^
    """
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
    """
    Field - or attribute - selector as part of a multi-element variable.

    Examples::

        .field
        .field^
    """
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
    """
    A multi-element variable - with one or more subscripts and fields.

    Examples::

        a.b.c
        a^.b[1].c
        a.b[SomeConstant].c^

    Where ``a`` is the "name"
    """
    #: The first part of the variable name.
    name: SimpleVariable
    #: This is unused (TODO / perhaps for compat elsewhere?)
    #: Dereference status is held on a per-element basis.
    dereferenced: bool
    #: The subscripts/fields that make up the multi-element variable.
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


class TypeInitializationBase:
    """
    Base class for type initializations.
    """

    @property
    def type_info(self) -> TypeInformation:
        """The base type name."""
        return TypeInformation.from_init(self)

    @property
    def base_type_name(self) -> Union[lark.Token, str]:
        """The base type name."""
        return self.type_info.base_type_name

    @property
    def full_type_name(self) -> Union[lark.Token, str]:
        """The full, qualified type name."""
        return self.type_info.full_type_name


class TypeSpecificationBase:
    """
    Base class for a specification of a type.

    Can specify a:

    1. Enumeration::

        ( 1, 1 ) INT
        TYPE_NAME     (TODO; ambiguous with 2)

    2. A simple or string type specification::

        TYPE_NAME
        STRING
        STRING[255]

    3. An indirect simple specification::

        POINTER TO TYPE_NAME
        REFERENCE TO TYPE_NAME
        REFERENCE TO POINTER TO TYPE_NAME

    4. An array specification::

        ARRAY [1..2] OF TypeName
        ARRAY [1..2] OF TypeName(1, 2)
    """
    @property
    def type_info(self) -> TypeInformation:
        """The base type name."""
        return TypeInformation.from_spec(self)

    @property
    def base_type_name(self) -> Union[lark.Token, str]:
        """The full type name."""
        return self.type_info.base_type_name

    @property
    def full_type_name(self) -> Union[lark.Token, str]:
        """The full type name."""
        return self.type_info.full_type_name


@dataclass
@_rule_handler("simple_spec_init")
class TypeInitialization(TypeInitializationBase):
    """
    A simple initialization specification of a type name.

    Example::

        TypeName := Value1
        STRING[100] := "value"
    """
    spec: Union[SimpleSpecification, IndirectSimpleSpecification]
    value: Optional[Expression]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return join_if(self.full_type_name, " := ", self.value)


@dataclass
@_rule_handler("simple_type_declaration")
class SimpleTypeDeclaration:
    """
    A declaration of a simple type.

    Examples::

        TypeName : INT
        TypeName : INT := 5
        TypeName : INT := 5 + 1 * (2)
        TypeName : REFERENCE TO INT
        TypeName : POINTER TO INT
        TypeName : POINTER TO POINTER TO INT
        TypeName : REFERENCE TO POINTER TO INT
        TypeName EXTENDS a.b : POINTER TO INT
    """
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
    """
    A string type declaration.

    Examples::
        TypeName : STRING
        TypeName : STRING := 'literal'
        TypeName : STRING[5]
        TypeName : STRING[100] := 'literal'
        TypeName : WSTRING[100] := "literal"
    """
    name: lark.Token
    string_type: StringTypeSpecification
    value: Optional[String]
    meta: Optional[Meta] = meta_field()

    @property
    def type_name(self) -> lark.Token:
        return self.string_type.type_name

    def __str__(self) -> str:
        type_and_value = join_if(self.string_type, " := ", self.value)
        return f"{self.name} : {type_and_value}"


@dataclass
@_rule_handler("string_type_specification")
class StringTypeSpecification(TypeSpecificationBase):
    """
    Specification of a string type.

    Examples::

        STRING(2_500_000)
        STRING(Param.iLower)
        STRING(Param.iLower * 2 + 10)
        STRING(Param.iLower / 2 + 10)

    Bracketed versions are also acceptable::

        STRING[2_500_000]
        STRING[Param.iLower]
    """
    type_name: lark.Token
    length: Optional[StringSpecLength] = None
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
class StringTypeInitialization(TypeInitializationBase):
    """
    Single or double-byte string specification.

    Examples::

        STRING := 'test'
        STRING(2_500_000) := 'test'
        STRING(Param.iLower) := 'test'

    Bracketed versions are also acceptable.
    """
    spec: StringTypeSpecification
    value: Optional[lark.Token]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        string_type: lark.Token,
        length: Optional[StringSpecLength] = None,
        value: Optional[lark.Token] = None,
    ) -> StringTypeInitialization:
        spec = StringTypeSpecification(string_type, length)
        return StringTypeInitialization(spec=spec, value=value)

    def __str__(self) -> str:
        return join_if(self.spec, " := ", self.value)


@dataclass
@as_tagged_union
class Subrange:
    """
    Subrange base class.

    May be a full or partial sub-range. Marked as a "tagged union" so that
    serialization will uniquely identify the Python class.
    """
    ...


@dataclass
class FullSubrange(Subrange):
    """
    A full subrange (i.e., asterisk ``*``).

    Example::

        Array[*]
              ^
    """
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return "*"


@dataclass
@_rule_handler("subrange")
class PartialSubrange(Subrange):
    """
    A partial subrange, including a start/stop element index.

    Examples::

        1..2
        iStart..iEnd
    """
    start: Expression
    stop: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.start}..{self.stop}"


@dataclass
@_rule_handler("subrange_specification")
class SubrangeSpecification(TypeSpecificationBase):
    """
    A subrange specification.

    Examples::

        INT (*)
        INT (1..2)
        TYPE_NAME         (TODO; overlap)
    """
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
class SubrangeTypeInitialization(TypeInitializationBase):
    """
    A subrange type initialization.

    Examples::

        INT (1..2) := 25
    """
    # TODO: coverage + examples?
    indirection: Optional[IndirectionType]
    spec: SubrangeSpecification
    value: Optional[Expression] = None
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        spec = join_if(self.indirection, " ", self.spec)
        if not self.value:
            return spec

        return f"{spec} := {self.value}"


@dataclass
@_rule_handler("subrange_type_declaration")
class SubrangeTypeDeclaration:
    """
    A subrange type declaration.

    Examples::

        TypeName : INT (1..2)
        TypeName : INT (*) := 1
    """
    name: lark.Token
    init: SubrangeTypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("enumerated_value")
class EnumeratedValue:
    """
    An enumerated value.

    Examples::

        IdentifierB
        IdentifierB := 1
        INT#IdentifierB
        INT#IdentifierB := 1
    """
    # TODO: coverage?
    type_name: Optional[lark.Token]
    name: lark.Token
    value: Optional[Union[Integer, FunctionCall, lark.Token]]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        name = join_if(self.type_name, "#", self.name)
        return join_if(name, " := ", self.value)


@dataclass
@_rule_handler("enumerated_specification")
class EnumeratedSpecification(TypeSpecificationBase):
    """
    An enumerated specification.

    Examples::

        (Value1, Value2 := 1)
        (Value1, Value2 := 1) INT
        INT
    """
    _implicit_type_default_: ClassVar[str] = "INT"
    type_name: Optional[lark.Token]
    values: Optional[List[EnumeratedValue]] = None
    meta: Optional[Meta] = meta_field()

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
class EnumeratedTypeInitialization(TypeInitializationBase):
    """
    Enumerated specification with initialization enumerated value.

    May be indirect (i.e., POINTER TO).

    Examples::

        (Value1, Value2 := 1) := IdentifierB
        (Value1, Value2 := 1) INT := IdentifierC
        INT := IdentifierD
    """
    # TODO coverage + double-check examples (doctest-like?)
    indirection: Optional[IndirectionType]
    spec: EnumeratedSpecification
    value: Optional[EnumeratedValue]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        spec = join_if(self.indirection, " ", self.spec)
        return join_if(spec, " := ", self.value)


@dataclass
@_rule_handler("enumerated_type_declaration", comments=True)
class EnumeratedTypeDeclaration:
    """
    An enumerated type declaration.

    Examples::

        TypeName : TypeName := Va
        TypeName : (Value1 := 1, Value2 := 2)
        TypeName : (Value1 := 1, Value2 := 2) INT := Value1
    """
    name: lark.Token
    init: EnumeratedTypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("non_generic_type_name")
class DataType:
    """
    A non-generic type name, or a data type name.

    May be indirect (e.g., POINTER TO).

    An elementary type name, a derived type name, or a general dotted
    identifier are valid for this.
    """
    # TODO: more grammar overlaps with dotted/simple names?
    indirection: Optional[IndirectionType]
    type_name: Union[lark.Token, StringTypeSpecification]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        if self.indirection and self.indirection.is_indirect:
            return f"{self.indirection} {self.type_name}"
        return f"{self.type_name}"


@dataclass
@_rule_handler("simple_specification")
class SimpleSpecification(TypeSpecificationBase):
    """
    A simple specification with just a type name (or a string type name).

    An elementary type name, a simple type name, or a general dotted
    identifier are valid for this.
    """
    type: Union[lark.Token, StringTypeSpecification]
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return str(self.type)


@dataclass
@_rule_handler("indirect_simple_specification")
class IndirectSimpleSpecification(TypeSpecificationBase):
    """
    A simple specification with the possibility of indirection.

    Examples::

        TypeName
        POINTER TO TypeName
        REFERENCE TO TypeName
        REFERENCE TO POINTER TO TypeName

    Initialization parameters such as these are parsed but otherwise ignored
    by TwinCAT::

        POINTER TO TypeName(1, 2)
        POINTER TO TypeName(1, 2, C := 4)
    """
    indirection: Optional[IndirectionType]
    type: SimpleSpecification
    init_parameters: Optional[List[InputParameterAssignment]]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        indirection: Optional[IndirectionType],
        type_: SimpleSpecification,
        init_parameters_tree: Optional[lark.Tree],
    ) -> IndirectSimpleSpecification:
        if init_parameters_tree is None:
            init_parameters = None
        else:
            init_parameters = typing.cast(
                List[InputParameterAssignment],
                list(init_parameters_tree.children)
            )
        return IndirectSimpleSpecification(
            indirection,
            type_,
            init_parameters,
        )

    def __str__(self) -> str:
        full_type = join_if(self.indirection, " ", self.type)
        if not self.init_parameters:
            return full_type

        initializers = ", ".join(
            str(init)
            for init in self.init_parameters
        )
        return f"{full_type}({initializers})"


# _array_spec_type
ArraySpecType = Union[
    DataType,
    "FunctionCall",
    "ObjectInitializerArray",
    "ArraySpecification",
    StringTypeSpecification,
]


@dataclass
@_rule_handler("array_specification")
class ArraySpecification(TypeSpecificationBase):
    """
    An array specification.

    Examples::

        ARRAY[*] OF TypeName
        ARRAY[1..2] OF Call(1, 2)
        ARRAY[1..2] OF Call(1, 2)
        ARRAY[1..5] OF Vec(SIZEOF(TestStruct), 0)
        ARRAY[1..5] OF STRING[10]
        ARRAY[1..5] OF STRING(Param.iLower)
    """
    subranges: List[Subrange]
    type: ArraySpecType
    meta: Optional[Meta] = meta_field()

    @property
    def base_type_name(self) -> Union[str, lark.Token]:
        """The base type name."""
        if isinstance(self.type, DataType):
            if isinstance(self.type.type_name, StringTypeSpecification):
                return str(self.type)
            return self.type.type_name
        if isinstance(self.type, ArraySpecification):
            return self.type.base_type_name
        if isinstance(self.type, StringTypeSpecification):
            return self.type.base_type_name
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
    """
    Initial value for an array element (potentialy repeated).

    The element itself may be an expression, a structure initialization, an
    enumerated value, or an array initialization.

    It may have a repeat value (``count``) as in::

        Repeat(Value)
        10(5)
        Repeat(5 + 3)
        INT#IdentifierB(5 + 3)
    """
    # NOTE: order is correct here; see rule array_initial_element_count
    # TODO: check enumerated value for count? specifically the := one
    element: ArrayInitialElementType
    count: Optional[Union[EnumeratedValue, Integer]] = None
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        if self.count is None:
            return f"{self.element}"
        return f"{self.count}({self.element})"


@_rule_handler("array_initial_element_count")
class _ArrayInitialElementCount:
    """
    An internal handler for array initial elements with repeat count
    values.
    """
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
    """
    Internal handler for array initialization with brackets.

    See also :class:`ArrayInitialization`
    """
    @staticmethod
    def from_lark(*elements: ArrayInitialElement) -> ArrayInitialization:
        return ArrayInitialization(list(elements), brackets=True)


@dataclass
@_rule_handler("bare_array_initialization")
class _BareArrayInitialization:
    """
    Internal handler for array initialization, without brackets

    See also :class:`ArrayInitialization`
    """
    @staticmethod
    def from_lark(*elements: ArrayInitialElement) -> ArrayInitialization:
        return ArrayInitialization(list(elements), brackets=False)


@dataclass
class ArrayInitialization:
    """
    Array initialization (bare or bracketed).

    Examples::

        [1, 2, 3]
        1, 2, 3
    """
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
    """
    Object initialization in array form.

    Examples::

      FB_Runner[(name := 'one'), (name := 'two')]
    """
    name: lark.Token
    initializers: List[StructureInitialization]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        function_block_type_name: lark.Token,
        *initializers: StructureInitialization
    ) -> ObjectInitializerArray:
        return ObjectInitializerArray(
            name=function_block_type_name,
            initializers=list(initializers)
        )

    def __str__(self) -> str:
        initializers = ", ".join(
            maybe_add_brackets(str(init), "()")
            for init in self.initializers
        )
        return f"{self.name}[{initializers}]"


@dataclass
@_rule_handler("array_spec_init")
class ArrayTypeInitialization(TypeInitializationBase):
    """
    Array specification and optional default (initialization) value.

    May be indirect (e.g., POINTER TO).

    Examples::

        ARRAY[*] OF TypeName
        ARRAY[1..2] OF Call(1, 2) := [1, 2]
        POINTER TO ARRAY[1..2] OF Call(1, 2)
    """
    indirection: Optional[IndirectionType]
    spec: ArraySpecification
    value: Optional[ArrayInitialization]
    meta: Optional[Meta] = meta_field()

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
    """
    Full declaration of an array type.

    Examples::

        ArrayType : ARRAY[*] OF TypeName
        ArrayType : ARRAY[1..2] OF Call(1, 2) := [1, 2]
        ArrayType : POINTER TO ARRAY[1..2] OF Call(1, 2)
        TypeName : ARRAY [1..2, 3..4] OF INT
        TypeName : ARRAY [1..2] OF INT := [1, 2]
        TypeName : ARRAY [1..2, 3..4] OF INT := [2(3), 3(4)]
        TypeName : ARRAY [1..2, 3..4] OF Tc.SomeType
        TypeName : ARRAY [1..2, 3..4] OF Tc.SomeType(someInput := 3)
        TypeName : ARRAY [1..2, 3..4] OF ARRAY [1..2] OF INT
        TypeName : ARRAY [1..2, 3..4] OF ARRAY [1..2] OF ARRAY [3..4] OF INT
    """
    name: lark.Token
    init: ArrayTypeInitialization
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("structure_type_declaration", comments=True)
class StructureTypeDeclaration:
    """
    Full structure type declaration, as part of a TYPE.

    Examples::

        TypeName EXTENDS Other.Type :
        STRUCT
            iValue : INT;
        END_STRUCT

        TypeName : POINTER TO
        STRUCT
            iValue : INT;
        END_STRUCT

        TypeName : POINTER TO
        STRUCT
            iValue : INT := 3 + 4;
            stTest : ST_Testing := (1, 2);
            eValue : E_Test := E_Test.ABC;
            arrValue : ARRAY [1..2] OF INT := [1, 2];
            arrValue1 : INT (1..2);
            arrValue1 : (Value1 := 1) INT;
            sValue : STRING := 'abc';
            iValue1 AT %I* : INT := 5;
            sValue1 : STRING[10] := 'test';
        END_STRUCT
    """
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
                    indent("\n".join(f"{decl};" for decl in self.declarations)),
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
    """
    Declaration line of a structure, typically with a single variable name.

    Excludes the trailing semicolon.

    Examples::

        iValue : INT := 3 + 4
        stTest : ST_Testing := (1, 2)
        eValue : E_Test := E_Test.ABC
        arrValue : ARRAY [1..2] OF INT := [1, 2]
        arrValue1 : INT (1..2)
        arrValue1 : (Value1 := 1) INT
        sValue : STRING := 'abc'
        iValue1 AT %I* : INT := 5
        sValue1 : STRING[10] := 'test'
        Timer1, Timer2, Timer3 : library.TPUDO
    """
    variables: List[DeclaredVariable]
    init: Union[
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
    def value(self) -> str:
        """The initialization value, if applicable."""
        return str(self.init.value)

    @property
    def base_type_name(self) -> Union[lark.Token, str]:
        """The base type name."""
        return self.init.base_type_name

    @property
    def full_type_name(self) -> Union[lark.Token, str]:
        """The full type name."""
        return self.init.full_type_name

    def __str__(self) -> str:
        variables = ", ".join(str(var) for var in self.variables)
        return f"{variables} : {self.init}"


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
    """
    Declaration of a single element of a union.

    Similar to a structure element, but not all types are supported and no
    initialization/default values are allowed.

    Examples::

        iValue : INT;
        arrValue : ARRAY [1..2] OF INT;
        arrValue1 : INT (1..2);
        arrValue1 : (Value1 := 1) INT;
        sValue : STRING;
        psValue1 : POINTER TO STRING[10];
    """
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
    """
    A full declaration of a UNION type, as part of a TYPE/END_TYPE block.

    Examples::

        UNION
            iVal : INT;
            aAsBytes : ARRAY [0..2] OF BYTE;
        END_UNION

        UNION
            iValue : INT;
            eValue : (iValue := 1, iValue2 := 2) INT;
        END_UNION
    """
    name: lark.Token
    extends: Optional[Extends]
    declarations: List[UnionElementDeclaration]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        name: lark.Token,
        extends: Optional[Extends],
        *decls: UnionElementDeclaration
    ):
        return UnionTypeDeclaration(
            name=name,
            extends=extends,
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

        definition = join_if(self.name, " ", self.extends)

        return "\n".join(
            line for line in (
                f"{definition} :",
                "UNION",
                decls,
                "END_UNION",
            )
            if line is not None
        )


@dataclass
@_rule_handler("initialized_structure")
class InitializedStructure(TypeInitializationBase):
    """
    A named initialized structure.

    Examples::

        ST_TypeName := (iValue := 0, bValue := TRUE)
    """
    name: lark.Token
    init: StructureInitialization
    meta: Optional[Meta] = meta_field()

    @property
    def value(self) -> str:
        """The initialization value (call)."""
        return str(self.init)

    def __str__(self) -> str:
        return f"{self.name} := {self.init}"


@dataclass
@_rule_handler("structure_initialization")
class StructureInitialization:
    """
    A structure initialization (i.e., default values) of one or more elements.

    Elements may be either positional or named.  Used in the following:

    1. Structure element initialization of default values::

        stStruct : ST_TypeName := (iValue := 0, bValue := TRUE)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    2. Function block declarations (fb_name_decl, fb_invocation_decl)::

        fbSample : FB_Sample(nInitParam := 1) := (nInput := 2, nMyProperty := 3)
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        fbSample : FB_Sample := (nInput := 2, nMyProperty := 3)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    3. Array object initializers (object_initializer_array)::

        runners : ARRAY[1..2] OF FB_Runner[(name := 'one'), (name := 'two')]
                                          [^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^]
    """
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
    """
    An initialization (default) value for a structure element.

    This may come in the form of::

        name := value

    or simply::

        value

    ``value`` may refer to an expression, an enumerated value, represent
    a whole array, or represent a nested structure.
    """
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
    """A unary - single operand - operation: ``NOT``, ``-``, or ``+``."""
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
    "expression_term"
)
class BinaryOperation(Expression):
    """
    A binary (i.e., two operand) operation.

    Examples::

        a + b
        a AND b
        a AND_THEN b
        a OR_ELSE b
        a := b
        a XOR b
        a = b
        -a * b
        a * 1.0

    Expressions may be nested in either the left or right operand.
    """
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
    """
    An expression with parentheses around it.

    Examples::

        (a * b)
        (1 + b)
    """
    expr: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"({self.expr})"


@dataclass
@_rule_handler("bracketed_expression")
class BracketedExpression(Expression):
    """
    An expression with square brackets around it.

    This is used exclusively in string length specifications.

    Examples::

        [a * b]
        [255]

    See also :class:`StringSpecLength`.
    """
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"[{self.expression}]"


@dataclass
@_rule_handler("string_spec_length")
class StringSpecLength:
    """
    The length of a defined string.

    The grammar makes a distinction between brackets and parentheses, though
    they appear to be functionally equivalent.

    Examples::

        [1]
        (1)
        [255]
    """

    length: Union[ParenthesizedExpression, BracketedExpression]

    def __str__(self) -> str:
        return str(self.length)


@dataclass
@_rule_handler("function_call")
class FunctionCall(Expression):
    """
    A function (function block, method, action, etc.) call.

    The return value may be dereferenced with a carat (``^``).

    Examples::

        A()^
        A(1, 2)
        A(1, 2, sName:='test', iOutput=>)
        A.B[1].C(1, 2)
    """
    #: The function name.
    name: SymbolicVariable
    #: Positional, naed, or output parameters.
    parameters: List[ParameterAssignment]
    #: Dereference the return value?
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
@_rule_handler("chained_function_call")
class ChainedFunctionCall(Expression):
    """
    A set of chained function (function block, method, action, etc.) calls.

    The return value may be dereferenced with a carat (``^``).

    Examples::

        A()^.B()
        A(1, 2).B().C()
        A(1, 2, sName:='test', iOutput=>).C().D()
        A.B[1].C(1, 2)
    """
    invocations: List[FunctionCall]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*invocations: FunctionCall) -> ChainedFunctionCall:
        return ChainedFunctionCall(
            invocations=list(invocations)
        )

    def __str__(self) -> str:
        return ".".join(str(invocation) for invocation in self.invocations)


@dataclass
@_rule_handler("var1")
class DeclaredVariable:
    """
    A single declared variable name and optional [direct or incomplete] location.

    Examples::

        iVar
        iVar AT %I*
        iVar AT %IX1.1
    """

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
    """
    Base class for a declaration of one or more variables with a type initialization.
    """
    variables: List[DeclaredVariable]
    init: InitDeclarationType
    meta: Optional[Meta]

    def __str__(self) -> str:
        variables = ", ".join(str(variable) for variable in self.variables)
        return f"{variables} : {self.init}"


@dataclass
@_rule_handler("var1_init_decl", comments=True)
class VariableOneInitDeclaration(InitDeclaration):
    """
    A declaration of one or more variables with a type, subrange, or enumerated
    type initialization.

    Examples::

        stVar1, stVar2 : (Value1, Value2)
        stVar1, stVar2 : (Value1 := 0, Value2 := 1)
        stVar1 : INT (1..2) := 25
        stVar1, stVar2 : TypeName := Value
        stVar1, stVar2 : (Value1 := 1, Value2 := 2)
        stVar1, stVar2 : (Value1 := 1, Value2 := 2) INT := Value1
    """
    variables: List[DeclaredVariable]
    init: Union[TypeInitialization, SubrangeTypeInitialization, EnumeratedTypeInitialization]
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler("array_var_init_decl", comments=True)
class ArrayVariableInitDeclaration(InitDeclaration):
    """
    A declaration of one or more variables with array type initialization and
    optional default (initialization) value.

    May be indirect (e.g., POINTER TO).

    Examples::

        aVal1, aVal2 : ARRAY[*] OF TypeName
        aVal1 : ARRAY[1..2] OF Call(1, 2) := [1, 2]
        aVal1 : POINTER TO ARRAY[1..2] OF Call(1, 2)
    """
    variables: List[DeclaredVariable]
    init: ArrayTypeInitialization
    meta: Optional[Meta] = meta_field()


@dataclass
@_rule_handler("structured_var_init_decl", comments=True)
class StructuredVariableInitDeclaration(InitDeclaration):
    """
    A declaration of one or more variables using a named initialized structure.

    Examples::

        stVar1 : ST_TypeName := (iValue := 0, bValue := TRUE)
        stVar1, stVar2 : ST_TypeName := (iValue  = 0, bValue := TRUE)
    """
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
    """
    A declaration of one or more variables using single/double byte strings,
    with an optinoal initialization value.

    Examples::

        sVar1 : STRING(2_500_000) := 'test1'
        sVar2, sVar3 : STRING(Param.iLower) := 'test2'
        sVar4, sVar5 : WSTRING(Param.iLower) := "test3"
    """
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
    """
    An edge declaration of one or more variables.

    Examples::

        iValue AT %IX1.1 : BOOL R_EDGE
        iValue : BOOL F_EDGE
    """
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
    """
    Base class for declarations of variables using function blocks.

    May either be by name (:class:`FunctionBlockNameDeclaration`) or invocation
    :class:`FunctionBlockInvocationDeclaration`). Marked as a "tagged union" so
    that serialization will uniquely identify the Python class.
    """
    ...


@dataclass
@_rule_handler("fb_name_decl", comments=True)
class FunctionBlockNameDeclaration(FunctionBlockDeclaration):
    """
    Base class for declarations of variables using function blocks by name.

    Examples::

        fbName1 : FB_Name
        fbName1 : FB_Name := (iValue := 0, bValue := TRUE)
    """
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
    """
    Base class for declarations of variables using function blocks by invocation.

    Examples::

        fbSample : FB_Sample(nInitParam := 1) := (nInput := 2, nMyProperty := 3)
    """
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
    """
    Base class for assigned parameters in function calls.

    May be either input parameters (positional or named ``name :=``) or output
    parameters (named as in ``name =>``, ``NOT name =>``). Marked as a "tagged
    union" so that serialization will uniquely identify the Python class.
    """
    ...


@dataclass
@_rule_handler("param_assignment", "input_param_assignment")
class InputParameterAssignment(ParameterAssignment):
    """
    An input parameter in a function call.

    May be a nameless positional parameter or a named one.

    Examples::

        name := value
    """
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
    """
    A named output parameter, which may be inverted.

    Examples::

        name => output
        NOT name => output2
        name =>
        NOT name =>
    """
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
    """
    Global variable specification; the part that comes before the
    initialization.

    Located (or incomplete located) specifications only apply to one variable,
    whereas simple specifications can have multiple variables.

    Examples::

        iValue1, iValue2
        iValue3 AT %I*
        iValue4 AT %IX1.1
    """
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
    """
    A declaration of one or more global variables: name and location
    specification and initialization type.

    Examples::

        fValue1 : INT;
        fValue2 : INT (0..10);
        fValue3 : (A, B);
        fValue4 : (A, B) DINT;
        fValue5 : ARRAY [1..10] OF INT;
        fValue6 : ARRAY [1..10] OF ARRAY [1..10] OF INT;
        fValue7 : FB_Test(1, 2, 3);
        fValue8 : FB_Test(A := 1, B := 2, C => 3);
        fValue9 : STRING[10] := 'abc';
    """
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
        return f"{self.spec} : {self.init};"


@dataclass
@_rule_handler("extends")
class Extends:
    """
    The "EXTENDS" portion of a function block, interface, structure, etc.

    Examples::

        EXTENDS stName
        EXTENDS FB_Name
    """

    name: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"EXTENDS {self.name}"


@dataclass
@_rule_handler("implements")
class Implements:
    """
    The "IMPLEMENTS" portion of a function block, indicating it implements
    one or more interfaces.

    Examples::

        IMPLEMENTS I_Interface1
        IMPLEMENTS I_Interface1, I_Interface2
    """

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
    """
    A full function block type declaration.

    A function block distinguishes itself from a regular function by having
    state and potentially having actions, methods, and properties. These
    additional parts are separate in this grammar (i.e., they do not appear
    within the FUNCTION_BLOCK itself).

    An implementation is optional, but ``END_FUNCTION_BLOCK`` is required.

    Examples::

        FUNCTION_BLOCK FB_EmptyFunctionBlock
        END_FUNCTION_BLOCK

        FUNCTION_BLOCK FB_Implementer IMPLEMENTS I_fbName
        END_FUNCTION_BLOCK

        FUNCTION_BLOCK ABSTRACT FB_Extender EXTENDS OtherFbName
        END_FUNCTION_BLOCK

        FUNCTION_BLOCK FB_WithVariables
        VAR_INPUT
            bExecute : BOOL;
        END_VAR
        VAR_OUTPUT
            iResult : INT;
        END_VAR
        END_FUNCTION_BLOCK
    """
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
    """
    A full function block type declaration, with nested variable declaration blocks.

    An implementation is optional, but ``END_FUNCTION`` is required.

    Examples::

        FUNCTION FuncName : INT
            VAR_INPUT
                iValue : INT := 0;
            END_VAR
            FuncName := iValue;
        END_FUNCTION
    """
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
    """
    A full program declaration, with nested variable declaration blocks.

    An implementation is optional, but ``END_PROGRAM`` is required.

    Examples::

        PROGRAM ProgramName
            VAR_INPUT
                iValue : INT;
            END_VAR
            VAR_ACCESS
                AccessName : SymbolicVariable : TypeName READ_WRITE;
            END_VAR
            iValue := iValue + 1;
        END_PROGRAM
    """
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


InterfaceVariableDeclarationBlock = Union[
    "InputDeclarations",
    "OutputDeclarations",
    "InputOutputDeclarations",
    "ExternalVariableDeclarations",
    "VariableDeclarations",
]


@dataclass
@_rule_handler("interface_declaration", comments=True)
class Interface:
    """
    A full interface declaration, with nested variable declaration blocks.

    An implementation is not allowed for interfaces, but ``END_INTERFACE`` is
    still required.

    Examples::

    """
    name: lark.Token
    extends: Optional[Extends]
    # TODO: want this to be tagged during serialization, so it's kept as
    # VariableDeclarationBlock.  More specifically it is
    # declarations: List[InterfaceVariableDeclarationBlock]
    declarations: List[VariableDeclarationBlock]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(
        name: lark.Token,
        # access: Optional[AccessSpecifier],
        extends: Optional[Extends],
        *decls: VariableDeclarationBlock,
    ) -> Interface:
        return Interface(
            name=name,
            extends=extends,
            declarations=list(decls),
        )

    def __str__(self) -> str:
        header = f"INTERFACE {self.name}"
        header = join_if(header, " ", self.extends)
        return "\n".join(
            line for line in
            (
                header,
                *[str(declaration) for declaration in self.declarations],
                "END_INTERFACE",
            )
            if line is not None
        )


@dataclass
@_rule_handler("action", comments=True)
class Action:
    """
    A full, named action declaration.

    Actions belong to function blocks. Actions may not contain variable blocks,
    but may contain an implementation.  Variable references are assumed to be
    from the local namespace (i.e., the owner function block) or in the global
    scope.

    Examples::

        ACTION ActName
        END_ACTION

        ACTION ActName
            iValue := iValue + 2;
        END_ACTION
    """
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
    """
    A full, named method declaration.

    Methods belong to function blocks. Methods may contain variable blocks
    and a return type, and may also contain an implementation.

    Examples::

        METHOD PRIVATE MethodName : ARRAY [1..2] OF INT
        END_METHOD

        METHOD MethodName : INT
            MethodName := 1;
        END_METHOD
    """
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
    """
    A named property declaration, which may pertain to a ``get`` or ``set``.

    Properties belong to function blocks. Properties may contain variable
    blocks and a return type, and may also contain an implementation.

    Examples::

        PROPERTY PropertyName : RETURNTYPE
            VAR_INPUT
                bExecute : BOOL;
            END_VAR
            VAR_OUTPUT
                iResult : INT;
            END_VAR
            iResult := 5;
            PropertyName := iResult + 1;
        END_PROPERTY

        PROPERTY PRIVATE PropertyName : ARRAY [1..2] OF INT
        END_PROPERTY
    """
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
    """
    Base class for variable declaration blocks.

    Marked as a "tagged union" so that serialization will uniquely identify the
    Python class.
    """
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
    """
    Variable declarations block (``VAR``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    Static variable declarations block (``VAR_STAT``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    Temporary variable declarations block (``VAR_TEMP``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    Declarations block for instance variables in methods (``VAR_INST``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    Declaration of a variable in a VAR block that is located.
    """
    # TODO examples
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
    """
    Located variable declarations block (``VAR``).

    May be annotated with attributes (see :class:`VariableAttributes`).

    All variables in this are expected to be located (e.g., ``AT %IX1.1``).
    """
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


#: var_spec in the grammar
IncompleteLocatedVariableSpecInit = Union[
    SimpleSpecification,
    SubrangeTypeInitialization,
    EnumeratedTypeInitialization,
    StringTypeSpecification,
]


@dataclass
@_rule_handler("incomplete_located_var_decl", comments=True)
class IncompleteLocatedVariableDeclaration:
    """
    A named, incomplete located variable declaration inside a variable block.
    """
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
    """
    Incomplete located variable declarations block (``VAR``).

    May be annotated with attributes (see :class:`VariableAttributes`).

    All variables in this are expected to have incomplete locations (e.g., just
    ``%I*``).
    """
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
    """
    A named, external variable declaration inside a variable block.
    """
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
    """
    A block of named, external variable declarations (``VAR_EXTERNAL``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    A block of named, input variable declarations (``VAR_INPUT``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    A block of named, output variable declarations (``VAR_OUTPUT``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
    """
    A block of named, input/output variable declarations (``VAR_IN_OUT``).

    May be annotated with attributes (see :class:`VariableAttributes`).
    """
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
@_rule_handler("program_access_decl", comments=True)
class AccessDeclaration:
    """
    A single, named program access declaration.

    Examples::

        AccessName : SymbolicVariable : TypeName READ_WRITE;
        AccessName1 : SymbolicVariable1 : TypeName1 READ_ONLY;
        AccessName2 : SymbolicVariable2 : TypeName2;
    """
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
@_rule_handler("program_access_decls", comments=True)
class AccessDeclarations(VariableDeclarationBlock):
    """
    A block of named, program access variable declarations (``VAR_ACCESS``).

    See Also
    --------
    :class:`AccessDeclaration`
    """
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
    """
    Global variable declarations block (``VAR_GLOBAL``).

    May be annotated with attributes (see :class:`GlobalVariableAttributes`).
    """
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
                *(indent(str(item)) for item in self.items),
                "END_VAR",
            )
        )


@as_tagged_union
class Statement:
    """
    Base class for all statements in a structured text implementation section.

    Marked as a "tagged union" so that serialization will uniquely identify the
    Python class.
    """


@_rule_handler("function_call_statement", comments=True)
class FunctionCallStatement(Statement, FunctionCall):
    """
    A function (function block, method, action, etc.) call as a statement.

    Examples::

        A(1, 2);
        A(1, 2, sName:='test', iOutput=>);
        A.B[1].C(1, 2);
    """

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
    """
    A chained set of function calls as a statement, in a "fluent" style.

    Examples::

        uut.dothis().andthenthis().andthenthat();
        uut.getPointerToStruct()^.dothis(A := 1).dothat(B := 2).done();
    """
    invocations: List[FunctionCall]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(chain: ChainedFunctionCall) -> ChainedFunctionCallStatement:
        return ChainedFunctionCallStatement(
            invocations=chain.invocations,
        )

    def __str__(self) -> str:
        invoc = ".".join(str(invocation) for invocation in self.invocations)
        return f"{invoc};"


@dataclass
@_rule_handler("else_if_clause", comments=True)
class ElseIfClause:
    """The else-if ``ELSIF`` part of an ``IF/ELSIF/ELSE/END_IF`` block."""
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
    """The ``ELSE`` part of an ``IF/ELSIF/ELSE/END_IF`` block."""
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
    """The ``IF`` part of an ``IF/ELSIF/ELSE/END_IF`` block."""
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
    """
    A single element of a ``CASE`` statement block.

    May contain one or more matches with corresponding statements. Matches
    may include subranges, integers, enumerated values, symbolic variables,
    bit strings, or boolean values.

    See Also
    --------
    :class:`CaseMatch`
    """
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
    """
    A switch-like ``CASE`` statement block.

    May contain one or more cases with corresponding statements, and a default
    ``ELSE`` clause.

    See Also
    --------
    :class:`CaseElement`
    :class:`ElseClause`
    """
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
    """
    A no-operation statement referring to a variable and nothing else.

    Distinguished from an action depending on if the context-sensitive
    name matches an action or a variable name.

    Note that blark does not handle this for you and may arbitrarily choose
    one or the other.

    Examples::

        variable;
    """
    variable: Variable
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable};"


@dataclass
@_rule_handler("set_statement", comments=True)
class SetStatement(Statement):
    """
    A "set" statement which conditionally sets a variable to ``TRUE``.

    Examples::

        bValue S= iValue > 5;
    """
    variable: SymbolicVariable
    op: lark.Token
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable} S= {self.expression};"


@dataclass
@_rule_handler("reference_assignment_statement", comments=True)
class ReferenceAssignmentStatement(Statement):
    """
    A reference assignment statement.

    Examples::

        refOne REF= refOtherOne;
    """
    variable: SymbolicVariable
    op: lark.Token
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable} REF= {self.expression};"


@dataclass
@_rule_handler("reset_statement")
class ResetStatement(Statement):
    """
    A "reset" statement which conditionally clears a variable to ``FALSE``.

    Examples::

        bValue R= iValue <= 5;
    """
    variable: SymbolicVariable
    op: lark.Token
    expression: Expression
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return f"{self.variable} R= {self.expression};"


@dataclass
@_rule_handler("exit_statement", comments=True)
class ExitStatement(Statement):
    """A statement used to exit a loop, ``EXIT``."""
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "EXIT;"


@dataclass
@_rule_handler("continue_statement", comments=True)
class ContinueStatement(Statement):
    """A statement used to jump to the top of a loop, ``CONTINUE``."""
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "CONTINUE;"


@dataclass
@_rule_handler("return_statement", comments=True)
class ReturnStatement(Statement):
    """
    A statement used to return from a function [block], ``RETURN``.

    No value is allowed to be returned with this statement.
    """
    meta: Optional[Meta] = meta_field()

    def __str__(self):
        return "RETURN;"


@dataclass
@_rule_handler("assignment_statement", comments=True)
class AssignmentStatement(Statement):
    """
    An assignment statement.

    Examples::

        iValue := 5;
        iValue1 := iValue2 := 6;
    """
    variables: List[Variable]
    expression: Expression
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args) -> AssignmentStatement:
        *variables_and_ops, expression = args
        variables = variables_and_ops[::2]
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
    """A beginning conditional loop statement, ``WHILE``."""
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
    """An ending conditional loop statement, ``REPEAT``."""
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
    """
    A loop with a control variable and a start, stop, and (optional) step value.

    Examples::

        FOR iIndex := 0 TO 10
        DO
            iValue := iIndex * 2;
        END_FOR

        FOR iIndex := (iValue - 5) TO (iValue + 5) BY 2
        DO
            arrArray[iIndex] := iIndex * 2;
        END_FOR
    """
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
@_rule_handler(
    "labeled_statement",
    "end_of_statement_list_label",
    comments=True,
)
class LabeledStatement(Statement):
    """
    A statement marked with a user-defined label.

    This is to support the "goto"-style ``JMP``.

    Examples::

        label1: A := 1;

        label2:
        IF iValue = 1 THEN
            A := 3;
        END_IF
    """
    label: lark.Token
    statement: Optional[Statement] = None
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        if self.statement is None:
            return f"{self.label} :"

        statement = str(self.statement)
        if statement.count("\n") > 1:
            # Multiline statement after label - put it on the next line
            return f"{self.label} :\n{statement}"
        # Single line statement after label - put it on the same line
        return f"{self.label} : {statement}"


@dataclass
@_rule_handler("jmp_statement", comments=True)
class JumpStatement(Statement):
    """
    This is the "goto"-style ``JMP``, which points at a label.

    Examples::

        JMP label;
    """
    label: lark.Token
    meta: Optional[Meta] = meta_field()

    def __str__(self) -> str:
        return f"JMP {self.label};"


@dataclass
@_rule_handler("statement_list", "case_element_statement_list")
class StatementList:
    """A list of statements, making up a structured text implementation."""
    statements: List[Statement]
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*statements: Statement) -> StatementList:
        return StatementList(
            statements=list(statements)
        )

    def __str__(self) -> str:
        return "\n".join(str(statement) for statement in self.statements)


# FunctionBlockBody = Union[
#     StatementList,
# ]

FunctionBlockBody = StatementList  # Only supported option, for now
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
    """
    A data type declaration, wrapping the other declaration types with
    ``TYPE``/``END_TYPE``.

    Access specifiers may be included.

    See Also
    --------
    :class:`AccessSpecifier`
    :class:`ArrayTypeDeclaration`
    :class:`StructureTypeDeclaration`
    :class:`StringTypeDeclaration`
    :class:`SimpleTypeDeclaration`
    :class:`SubrangeTypeDeclaration`
    :class:`EnumeratedTypeDeclaration`
    :class:`UnionTypeDeclaration`
    """
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
    Property,
    GlobalVariableDeclarations,
]


@dataclass
@_rule_handler("iec_source")
class SourceCode:
    """
    Top-level source code item.

    May contain zero or more of the following as items:

    * :class:`DataTypeDeclaration`
    * :class:`Function`
    * :class:`FunctionBlock`
    * :class:`Action`
    * :class:`Method`
    * :class:`Program`
    * :class:`Property`
    * :class:`GlobalVariableDeclarations`
    """
    items: List[SourceCodeItem]
    filename: Optional[pathlib.Path] = None
    raw_source: Optional[str] = None
    line_map: Optional[Dict[int, int]] = None
    meta: Optional[Meta] = meta_field()

    @staticmethod
    def from_lark(*args: SourceCodeItem) -> SourceCode:
        return SourceCode(list(args))

    def range_from_file_lines(self, start: int, end: int) -> list[str]:
        if not self.raw_source:
            return []

        code_lines = self.raw_source.split("\n")  # not splitlines()
        if not self.line_map:
            return code_lines[start - 1: end]

        line_map = {
            raw_line: file_line for (file_line, raw_line) in self.line_map.items()
        }
        return code_lines[line_map[start] - 1: line_map[end]]

    def __str__(self):
        return "\n".join(str(item) for item in self.items)


@dataclass
class ExtendedSourceCode(SourceCode):
    """
    Top-level source code item - extended to include the possibility of
    standalone implementation details (i.e., statement lists).

    See Also
    --------
    :class:`SourceCodeItem`
    :class:`StatementList`
    """

    items: List[Union[SourceCodeItem, StatementList]]


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

    def transform(self, tree: lark.Tree, *, line_map: Optional[dict[int, int]] = None):
        if line_map is not None:
            tree = rebuild_lark_tree_with_line_map(
                tree,
                code_line_to_file_line=line_map,
            )

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
    def struct_var1_list(self, *items: DeclaredVariable) -> List[DeclaredVariable]:
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


def transform(
    source_code: str,
    tree: lark.Tree,
    comments: Optional[list[lark.Token]] = None,
    line_map: Optional[dict[int, int]] = None,
    filename: Optional[pathlib.Path] = None,
) -> SourceCode:
    """
    Transform a ``lark.Tree`` into dataclasses.

    Parameters
    ----------
    source_code : str
        The plain source code.
    tree : lark.Tree
        The parse tree from lark.
    comments : list[lark.Token], optional
        A list of pre-processed comments.
    line_map : dict[int, int], optional
        A map of lines from ``source_code`` to file lines.
    filename : pathlib.Path, optional
        The file associated with the source code.

    Returns
    -------
    SourceCode
    """
    transformer = GrammarTransformer(
        comments=list(comments or []),
        fn=filename,
        source_code=source_code,
    )
    transformed = transformer.transform(tree, line_map=line_map)

    if isinstance(transformed, SourceCode):
        return transformed

    # TODO: this is for custom starting points and ignores that 'transformed'
    # may not be a typical "SourceCodeItem". Goal is just returning a
    # consistent SourceCode instance
    return SourceCode(
        items=[transformed],
        filename=filename,
        raw_source=source_code,
        line_map=line_map,
        meta=transformed.meta,
    )


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
    Expression,
    StructureInitialization,
    EnumeratedValue,
    ArrayInitialization,
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
