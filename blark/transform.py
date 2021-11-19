from __future__ import annotations

import enum
import functools
import inspect
import textwrap
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import (Any, Callable, ClassVar, Dict, Generator, List, Optional,
                    Tuple, Type, TypeVar, Union)

import lark

_rule_to_class = {}
_class_handlers = {}
_comment_consumers = []

T = TypeVar("T")
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


def indent_if(value: Optional[Any], indent: str = INDENT) -> Optional[str]:
    """Indented {value} if not None."""
    if value is not None:
        return textwrap.indent(str(value), indent)
    return None


def _commented(meta: lark.tree.Meta, item: Any, indent: str = "", suffix="") -> str:
    comments = getattr(meta, "comments", None)
    if not comments:
        return f"{indent}{item}{suffix}"

    block = "\n".join((*comments, f"{item}{suffix}"))
    return textwrap.indent(block, prefix=indent)


def _commented_block(func):
    @functools.wraps(func)
    def wrapped(self):
        return _commented(
            self.meta,
            func(self)
        )
    return wrapped


def _get_default_instantiator(cls: type):
    def instantiator(*args):
        return cls(*args)

    return instantiator


def _rule_handler(
    *rules: Union[str, List[str]],
    comments: bool = False
) -> Callable[[Type[T]], Type[T]]:
    """Decorator - the wrapped class will handle the provided rules."""
    def wrapper(cls: Type[T]) -> Type[T]:
        if not hasattr(cls, "from_lark"):
            cls.from_lark = _get_default_instantiator(cls)

        for rule in rules:
            handler = _rule_to_class.get(rule, None)
            if handler is not None:
                raise ValueError(f"Handler already specified for: {rule} ({handler})")

            _rule_to_class[rule] = cls
            _class_handlers[rule] = cls.from_lark

        if comments:
            # Mark ``cls`` as one that consumes comments when stringifying code.
            _comment_consumers.append(cls)
        return cls

    return wrapper


class Literal:
    """Literal value."""

    value: Any

    def __str__(self) -> str:
        return str(self.value)


Constant = Literal  # an alias for now


@dataclass
@_rule_handler("integer_literal")
class Integer(Literal):
    """Integer literal value."""

    value: lark.Token
    type: Optional[lark.Token] = None
    base: ClassVar[int] = 10

    @staticmethod
    def from_lark(
        type_name: Optional[lark.Token],
        value: Union[Integer, lark.Token],
        *,
        base: int = 10,
    ) -> Integer:
        if isinstance(value, Integer):
            # Adding type information; wrap Integer
            value.type = type_name
            return value
        cls = _base_to_integer_class[base]
        return cls(
            type=type_name,
            value=value,
        )

    def __str__(self) -> str:
        value = f"{self.base}#{self.value}" if self.base != 10 else str(self.value)
        if self.type:
            return f"{self.type}#{value}"
        return value


@dataclass
@_rule_handler("binary_integer")
class BinaryInteger(Integer):
    base: ClassVar[int] = 2

    @classmethod
    def from_lark(
        cls, value: Union[Integer, lark.Token],
    ) -> BinaryInteger:
        return super().from_lark(None, value, base=2)


@dataclass
@_rule_handler("octal_integer")
class OctalInteger(Integer):
    base: ClassVar[int] = 8

    @classmethod
    def from_lark(
        cls, value: Union[Integer, lark.Token],
    ) -> Integer:
        return super().from_lark(None, value, base=8)


@dataclass
@_rule_handler("hex_integer")
class HexInteger(Integer):
    base: ClassVar[int] = 16

    @classmethod
    def from_lark(
        cls, value: Union[Integer, lark.Token],
    ) -> Integer:
        return super().from_lark(None, value, base=16)


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
    type: Optional[lark.Token] = None

    @staticmethod
    def from_lark(type_name: Optional[lark.Token], value: lark.Token) -> Real:
        return Real(type=type_name, value=value)

    def __str__(self) -> str:
        if self.type:
            return f"{self.type}#{self.value}"
        return str(self.value)


@dataclass
@_rule_handler("bit_string_literal")
class BitString(Literal):
    """Bit string literal value."""

    type: Optional[lark.Token]
    value: lark.Token
    base: ClassVar[int] = 10

    @classmethod
    def from_lark(cls, type: Optional[lark.Token], value: lark.Token):
        return cls(type, value)

    def __str__(self) -> str:
        value = f"{self.base}#{self.value}" if self.base != 10 else str(self.value)
        if self.type:
            return f"{self.type}#{value}"
        return value


@dataclass
@_rule_handler("binary_bit_string_literal")
class BinaryBitString(BitString):
    """Binary bit string literal value."""
    base: ClassVar[int] = 2


@dataclass
@_rule_handler("octal_bit_string_literal")
class OctalBitString(BitString):
    """Octal bit string literal value."""
    base: ClassVar[int] = 8


@dataclass
@_rule_handler("hex_bit_string_literal")
class HexBitString(BitString):
    """Hex bit string literal value."""
    base: ClassVar[int] = 16


@dataclass
class Boolean(Literal):
    """Boolean literal value."""

    value: lark.Token

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

    @staticmethod
    def from_lark(interval: lark.Tree) -> Duration:
        kwargs = {tree.data: tree.children[0] for tree in interval.iter_subtrees()}

        return Duration(**kwargs)

    @property
    def value(self) -> str:
        """The duration value."""
        return "".join(
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
@_rule_handler("time_of_day")
class TimeOfDay(Literal):
    """Time of day literal value."""
    hour: lark.Token
    minute: lark.Token
    second: lark.Token

    @property
    def value(self) -> str:
        """The time of day value."""
        return f"{self.hour}:{self.minute}:{self.second}"

    def __str__(self):
        return f"TIME_OF_DAY#{self.value}"


@dataclass
@_rule_handler("date")
class Date(Literal):
    """Date literal value."""

    year: lark.Token
    month: lark.Token
    day: lark.Token

    @property
    def value(self) -> str:
        """The time of day value."""
        return f"{self.year}-{self.month}-{self.day}"

    def __str__(self):
        return f"DATE#{self.value}"


@dataclass
@_rule_handler("date_and_time")
class DateTime(Literal):
    """Date and time literal value."""

    date: Date
    time: TimeOfDay

    @staticmethod
    def from_lark(
        year: lark.Token,
        month: lark.Token,
        day: lark.Token,
        hour: lark.Token,
        minute: lark.Token,
        second: lark.Token,
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
@_rule_handler("string_literal")
class String(Literal):
    """String literal value."""
    value: lark.Token


@dataclass
class Expression:
    ...


@dataclass
class Variable(Expression):
    ...


@_rule_handler(
    "method_access",
    "property_access",
)
class MethodAccess(enum.Flag):
    public = enum.auto()
    private = enum.auto()
    abstract = enum.auto()
    protected = enum.auto()
    internal = enum.auto()
    final = enum.auto()

    @staticmethod
    def from_lark(token: lark.Token, *tokens: List[lark.Token]) -> MethodAccess:
        result = MethodAccess[token.lower()]
        for token in tokens:
            result |= MethodAccess[token.lower()]
        return result

    def __str__(self):
        return " ".join(
            option.name.upper()
            for option in MethodAccess
            if option in self
        )


@_rule_handler(
    "indirection_type",
    "pointer_type",
)
class IndirectionType(Enum):
    """Indirect access through a pointer or reference."""
    none = enum.auto()
    pointer = enum.auto()
    reference = enum.auto()

    @staticmethod
    def from_lark(token: Optional[lark.Token]) -> IndirectionType:
        return {
            "NONE": IndirectionType.none,
            "POINTER TO": IndirectionType.pointer,
            "REFERENCE TO": IndirectionType.reference,
        }[str(token).upper()]

    def __str__(self):
        return {
            IndirectionType.none: "",
            IndirectionType.pointer: "POINTER TO",
            IndirectionType.reference: "REFERENCE TO",
        }[self]


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


class VariableSizePrefix(str, Enum):
    bit = "X"
    byte = "B"
    word_16 = "W"
    dword_32 = "D"
    lword_64 = "L"


@dataclass
@_rule_handler("direct_variable")
class DirectVariable(Variable):
    location_prefix: VariableLocationPrefix
    location: lark.Token
    size_prefix: VariableSizePrefix
    bits: Optional[List[lark.Token]] = None

    @staticmethod
    def from_lark(
        location_prefix: lark.Token,
        size_prefix: Optional[VariableSizePrefix],
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
        bits = ".".join([""] + self.bits) if self.bits else ""
        return f"%{self.location_prefix}{self.size_prefix}{self.location}{bits}"


@dataclass
@_rule_handler("location")
class Location(DirectVariable):
    @staticmethod
    def from_lark(var: DirectVariable):
        return Location(
            var.location_prefix,
            var.location,
            var.size_prefix,
            var.bits,
        )

    def __str__(self) -> str:
        direct_loc = super().__str__()
        return f"AT {direct_loc}"


@dataclass
@_rule_handler("variable_name")
class SymbolicVariable(Variable):
    name: lark.Token
    dereferenced: bool

    @staticmethod
    def from_lark(identifier: lark.Token, dereferenced: Optional[lark.Token]):
        return SymbolicVariable(
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
    field: lark.Token
    dereferenced: bool

    @staticmethod
    def from_lark(dereferenced: Optional[lark.Token], field: lark.Token):
        return FieldSelector(
            field=field,
            dereferenced=dereferenced is not None
        )

    def __str__(self) -> str:
        return f"^.{self.field}" if self.dereferenced else f".{self.field}"


@dataclass
@_rule_handler("multi_element_variable")
class MultiElementVariable(SymbolicVariable):
    elements: List[Union[SubscriptList, FieldSelector]]

    @staticmethod
    def from_lark(variable_name, *subscript_or_field):
        if not subscript_or_field:
            return SymbolicVariable(
                name=variable_name,
                dereferenced=False
            )
        return MultiElementVariable(
            name=variable_name,
            elements=list(subscript_or_field),
            dereferenced=False,
        )

    def __str__(self) -> str:
        return "".join(str(part) for part in (self.name, *self.elements))


@dataclass
@_rule_handler("simple_spec_init")
class TypeInitialization:
    indirection: Optional[IndirectionType]
    spec: Optional[lark.Token]
    value: Optional[Expression]

    def __str__(self) -> str:
        type_ = join_if(self.indirection, " ", self.spec)
        return join_if(type_, " := ", self.value)


@dataclass
@_rule_handler("simple_type_declaration")
class SimpleTypeDeclaration:
    name: lark.Token
    extends: Optional[Extends]
    init: TypeInitialization

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
    value: lark.Token

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

    def __str__(self) -> str:
        return join_if(self.type_name, "", self.length)


@dataclass
@_rule_handler(
    "single_byte_string_spec",
    "double_byte_string_spec",
)
class StringTypeInitialization:
    spec: StringTypeSpecification
    value: Optional[lark.Token]

    @staticmethod
    def from_lark(
        *args: lark.Token,
    ) -> StringTypeInitialization:
        string_type, length, *value = args
        spec = StringTypeSpecification(string_type, length)
        if len(value):
            if len(value) == 1:  # lark 0.12.0
                value, = value
            else:  # lark 1.0
                _, value = value
        else:
            value = None
        return StringTypeInitialization(spec, value)

    def __str__(self) -> str:
        return join_if(self.spec, " := ", self.value)


class Subrange:
    ...


@dataclass
class FullSubrange(Subrange):
    def __str__(self) -> str:
        return "*"


@dataclass
@_rule_handler("subrange")
class PartialSubrange:
    start: Expression
    stop: Expression

    def __str__(self) -> str:
        return f"{self.start}..{self.stop}"


@dataclass
@_rule_handler("subrange_specification")
class SubrangeSpecification:
    type_name: lark.Token
    subrange: Optional[Subrange] = None

    def __str__(self) -> str:
        if self.subrange:
            return f"{self.type_name} ({self.subrange})"
        return f"{self.type_name}"


@dataclass
@_rule_handler("subrange_spec_init")
class SubrangeTypeInitialization:
    indirection: Optional[IndirectionType]
    spec: SubrangeSpecification
    value: Optional[Expression] = None

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

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("enumerated_value")
class EnumeratedValue:
    type_name: Optional[lark.Token]
    name: lark.Token
    value: Optional[Union[Integer, lark.Token]]

    def __str__(self) -> str:
        name = join_if(self.type_name, "#", self.name)
        return join_if(name, " := ", self.value)


@dataclass
@_rule_handler("enumerated_specification")
class EnumeratedSpecification:
    type_name: Optional[lark.Token]
    values: Optional[List[EnumeratedValue]] = None

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
    value: Optional[Expression]

    def __str__(self) -> str:
        spec = join_if(self.indirection, " ", self.spec)
        return join_if(spec, " := ", self.value)


@dataclass
@_rule_handler("enumerated_type_declaration", comments=True)
class EnumeratedTypeDeclaration:
    name: lark.Token
    init: EnumeratedTypeInitialization

    @_commented_block
    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("non_generic_type_name")
class DataType:
    indirection: Optional[IndirectionType]
    type_name: lark.Token

    def __str__(self) -> str:
        if self.indirection and self.indirection != IndirectionType.none:
            return f"{self.indirection} {self.type_name}"
        return f"{self.type_name}"


@dataclass
@_rule_handler("array_specification")
class ArraySpecification:
    type_name: DataType
    subranges: List[Subrange]

    @staticmethod
    def from_lark(*args):
        *subranges, type_name = args
        return ArraySpecification(type_name=type_name, subranges=subranges)

    def __str__(self) -> str:
        subranges = ", ".join(str(subrange) for subrange in self.subranges)
        return f"ARRAY [{subranges}] OF {self.type_name}"


ArrayInitialElementType = Union[
    Constant,
    "StructureInitialization",
    EnumeratedValue,
]


@dataclass
@_rule_handler("array_initial_element")
class ArrayInitialElement:
    element: ArrayInitialElementType

    def __str__(self) -> str:
        return f"{self.element}"


@dataclass
@_rule_handler("array_initial_element_count")
class ArrayInitialElementCount:
    count: Union[EnumeratedValue, Integer]
    element: ArrayInitialElementType

    def __str__(self) -> str:
        return f"{self.count}({self.element})"


@dataclass
@_rule_handler("array_initialization")
class ArrayInitialization:
    elements: List[ArrayInitialElement]

    @staticmethod
    def from_lark(*elements: ArrayInitialElement):
        return ArrayInitialization(list(elements))

    def __str__(self) -> str:
        elements = ", ".join(str(element) for element in self.elements)
        return f"[{elements}]"


@dataclass
@_rule_handler("array_spec_init")
class ArrayTypeInitialization:
    indirection: Optional[IndirectionType]
    spec: ArraySpecification
    value: Optional[ArrayInitialization]

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

    @_commented_block
    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("structure_type_declaration", comments=True)
class StructureTypeDeclaration:
    name: lark.Token
    extends: Optional[lark.Token]
    indirection: Optional[IndirectionType]
    declarations: List[StructureElementDeclaration]

    @staticmethod
    def from_lark(
        name: lark.Token,
        extends: Optional[lark.Token],
        indirection: Optional[IndirectionType],
        *declarations: List[StructureElementDeclaration],
    ):
        return StructureTypeDeclaration(
            name, extends, indirection, declarations
        )

    @_commented_block
    def __str__(self) -> str:
        if self.declarations:
            body = "\n".join(
                (
                    "STRUCT",
                    textwrap.indent(
                        "\n".join(str(decl) for decl in self.declarations),
                        prefix=INDENT
                    ),
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
    ]

    @_commented_block
    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init};"


@dataclass
@_rule_handler("initialized_structure")
class InitializedStructure:
    name: lark.Token
    init: StructureInitialization

    def __str__(self) -> str:
        return f"{self.name} := {self.init}"


@dataclass
@_rule_handler("structure_initialization")
class StructureInitialization:
    elements: List[StructureElementInitialization]

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
@_rule_handler("initialized_structure_type_declaration", comments=True)
class InitializedStructureTypeDeclaration:
    name: lark.Token
    extends: Optional[lark.Token]
    init: StructureInitialization

    @_commented_block
    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("unary_expression")
class UnaryOperation(Expression):
    op: lark.Token
    expr: Expression

    @staticmethod
    def from_lark(*args):
        if len(args) == 1:
            constant, = args
            return constant

        operator, expr = args
        if not operator:
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

    @staticmethod
    def from_lark(left: Expression, *operator_and_expr: Union[lark.Token, Expression]):
        if not operator_and_expr:
            return left

        def get_operator_and_expr() -> Generator[Tuple[lark.Token, Expression], None, None]:
            operators = operator_and_expr[::2]
            expressions = operator_and_expr[1::2]
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

    def __str__(self) -> str:
        return f"({self.expr})"


@dataclass
@_rule_handler("function_call")
class FunctionCall(Expression):
    name: SymbolicVariable
    parameters: List[ParameterAssignment]

    @staticmethod
    def from_lark(
        name: lark.Token,
        *parameters: ParameterAssignment
    ) -> FunctionCall:
        return FunctionCall(
            name=name,
            parameters=list(parameters)
        )

    def __str__(self) -> str:
        parameters = ", ".join(str(param) for param in self.parameters)
        return f"{self.name}({parameters})"


@dataclass
@_rule_handler("var1")
class DeclaredVariable:
    # Alternate name: VariableWithLocation? MaybeLocatedVariable?
    variable: SymbolicVariable
    location: Optional[Union[IncompleteLocation, Location]]

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
class InitDeclaration:
    variables: List[DeclaredVariable]

    @_commented_block
    def __str__(self) -> str:
        variables = ", ".join(str(variable) for variable in self.variables)
        return f"{variables} : {self.init}"


@dataclass
@_rule_handler("var1_init_decl", comments=True)
class VariableOneInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    init: Union[TypeInitialization, SubrangeTypeInitialization, EnumeratedTypeInitialization]


@dataclass
@_rule_handler("array_var_init_decl", comments=True)
class ArrayVariableInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    init: ArrayTypeInitialization


@dataclass
@_rule_handler("structured_var_init_decl", comments=True)
class StructuredVariableInitDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    init: InitializedStructure


@dataclass
@_rule_handler(
    "single_byte_string_var_declaration",
    "double_byte_string_var_declaration",
    comments=True
)
class StringVariableInitDeclaration(InitDeclaration):
    spec: StringTypeSpecification
    value: lark.Token

    @staticmethod
    def from_lark(variables: List[DeclaredVariable], string_info: StringTypeInitialization):
        return StringVariableInitDeclaration(
            variables=variables,
            spec=string_info.spec,
            value=string_info.value,
        )

    @property
    def init(self) -> str:
        return join_if(self.spec, " := ", self.value)


@dataclass
@_rule_handler("edge_declaration", comments=True)
class EdgeDeclaration(InitDeclaration):
    variables: List[DeclaredVariable]
    edge: lark.Token

    @property
    def init(self) -> str:
        return f"BOOL {self.edge}"


class FunctionBlockDeclaration:
    ...


@dataclass
@_rule_handler("fb_name_decl", comments=True)
class FunctionBlockNameDeclaration(FunctionBlockDeclaration):
    variables: List[lark.Token]
    spec: lark.Token
    init: Optional[StructureInitialization] = None

    @_commented_block
    def __str__(self) -> str:
        variables = ", ".join(self.variables)
        name_and_type = f"{variables} : {self.spec}"
        return join_if(name_and_type, " := ", self.init)


@dataclass
@_rule_handler("fb_invocation_decl", comments=True)
class FunctionBlockInvocationDeclaration(FunctionBlockDeclaration):
    variables: List[lark.Token]
    init: FunctionBlockInvocation

    @_commented_block
    def __str__(self) -> str:
        variables = ", ".join(self.variables)
        return f"{variables} : {self.init}"


@dataclass
@_rule_handler("param_assignment")
class ParameterAssignment:
    name: Optional[lark.Token]
    value: Optional[Expression]

    @staticmethod
    def from_lark(*args) -> ParameterAssignment:
        if len(args) == 1:
            value, = args
            name = None
        else:
            name, value = args
        return ParameterAssignment(name, value)

    def __str__(self) -> str:
        return join_if(self.name, " := ", self.value)


@dataclass
@_rule_handler("output_parameter_assignment")
class OutputParameterAssignment(ParameterAssignment):
    inverted: bool

    @staticmethod
    def from_lark(
        inverted: Optional[lark.Token],
        name: lark.Token,
        value: Expression,
    ) -> ParameterAssignment:
        return OutputParameterAssignment(name, value, inverted is not None)

    def __str__(self) -> str:
        prefix = "NOT " if self.inverted else ""
        return prefix + join_if(self.name, " => ", self.value)


@dataclass
@_rule_handler("fb_invocation")
class FunctionBlockInvocation:
    name: lark.Token
    parameters: List[ParameterAssignment]

    @staticmethod
    def from_lark(
        name: lark.Token,
        *parameters: ParameterAssignment
    ) -> FunctionBlockInvocation:
        return FunctionBlockInvocation(
            name=name,
            parameters=list(parameters)
        )

    def __str__(self) -> str:
        parameters = ", ".join(str(param) for param in self.parameters)
        return f"{self.name}({parameters})"


@dataclass
@_rule_handler("global_var_spec")
class GlobalVariableSpec:
    variables: List[lark.Token]
    location: Optional[Union[Location, IncompleteLocation]]

    @staticmethod
    def from_lark(
        name_or_names: Union[lark.Token, lark.Tree],
        location: Optional[Union[Location, IncompleteLocation]] = None
    ) -> GlobalVariableSpec:
        if location is None:
            return GlobalVariableSpec(
                variables=name_or_names.children,
                location=None
            )

        return GlobalVariableSpec(
            variables=[name_or_names],
            location=location
        )

    def __str__(self) -> str:
        if not self.location:
            return ", ".join(self.variables)
        return f"{self.variables[0]} : {self.location}"


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
    init: Union[
        LocatedVariableSpecInit,
        FunctionBlockInvocation,
        lark.Token  # FB type name
    ]

    @_commented_block
    def __str__(self) -> str:
        return f"{self.spec} : {self.init}"


@dataclass
@_rule_handler("extends")
class Extends:
    name: lark.Token

    def __str__(self) -> str:
        return f"EXTENDS {self.name}"


@dataclass
@_rule_handler("function_block_body", comments=True)
class FunctionBlockBody:
    source: Union[
        StatementList,
        InstructionList,
        # SfcNetwork
    ]

    @_commented_block
    def __str__(self) -> str:
        return str(self.source)


@dataclass
@_rule_handler("function_block_type_declaration", comments=True)
class FunctionBlock:
    name: lark.Token
    abstract: bool
    extends: Optional[Extends]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBlockBody]

    @staticmethod
    def from_lark(
        abstract: Optional[lark.Token],
        derived_name: lark.Token,
        extends: Extends,
        *args
    ) -> FunctionBlock:
        *declarations, body = args
        return FunctionBlock(
            name=derived_name,
            abstract=abstract is not None,
            extends=extends,
            declarations=list(declarations),
            body=body,
        )

    @_commented_block
    def __str__(self) -> str:
        abstract = "ABSTRACT " if self.abstract else ""
        return "\n".join(
            line for line in
            (
                join_if(f"FUNCTION_BLOCK {abstract}{self.name}", " ", self.extends),
                *[str(declaration) for declaration in self.declarations],
                indent_if(self.body),
                "END_FUNCTION_BLOCK",
            )
            if line is not None
        )


@dataclass
@_rule_handler("function_declaration", comments=True)
class Function:
    name: lark.Token
    return_type: Optional[lark.Token]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBlockBody]

    @staticmethod
    def from_lark(
        name: lark.Token,
        return_type: lark.Token,
        declarations: Optional[lark.Tree],
        body: Optional[FunctionBlockBody]
    ) -> Function:
        return Function(
            name=name,
            return_type=return_type,
            declarations=declarations.children if declarations else [],
            body=body,
        )

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            line for line in
            (
                f"FUNCTION {self.name} : {self.return_type}",
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
    body: Optional[FunctionBlockBody]

    @staticmethod
    def from_lark(
        name: lark.Token,
        declarations: Optional[lark.Tree],
        body: Optional[FunctionBlockBody]
    ) -> Program:
        return Program(
            name=name,
            declarations=declarations.children if declarations else [],
            body=body,
        )

    @_commented_block
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


class Action:
    ...


@dataclass
@_rule_handler("action", comments=True)
class NamedAction(Action):
    name: lark.Token
    body: Optional[FunctionBlockBody]

    @_commented_block
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
@_rule_handler("entry_action", comments=True)
class EntryAction(Action):
    body: Optional[FunctionBlockBody]

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            line for line in
            (
                "ENTRY_ACTION",
                indent_if(self.body),
                "END_ACTION",
            )
            if line is not None
        )


@dataclass
@_rule_handler("exit_action", comments=True)
class ExitAction(Action):
    body: Optional[FunctionBlockBody]

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            line for line in
            (
                "EXIT_ACTION",
                indent_if(self.body),
                "END_ACTION",
            )
            if line is not None
        )


@dataclass
@_rule_handler("function_block_method_declaration", comments=True)
class Method:
    access: Optional[MethodAccess]
    name: lark.Token
    return_type: Optional[LocatedVariableSpecInit]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBlockBody]

    @staticmethod
    def from_lark(
        access: Optional[MethodAccess],
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

    @_commented_block
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
    access: Optional[MethodAccess]
    name: lark.Token
    return_type: Optional[LocatedVariableSpecInit]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBlockBody]

    @staticmethod
    def from_lark(
        access: Optional[MethodAccess],
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

    @_commented_block
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


@dataclass
class VariableDeclarationBlock:
    ...


VariableInitDeclaration = Union[
    ArrayVariableInitDeclaration,
    StringVariableInitDeclaration,
    VariableOneInitDeclaration,
    FunctionBlockDeclaration,
    # EdgeDeclaration,
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
# FunctionBlockDeclarations = Union[
#     ...
# ]


@dataclass
@_rule_handler("var_declarations", comments=True)
class VariableDeclarations(VariableDeclarationBlock):
    config: Optional[lark.Token]
    items: List[VariableInitDeclaration]

    @staticmethod
    def from_lark(config: Optional[lark.Token], items: lark.Tree) -> VariableDeclarations:
        return VariableDeclarations(
            config=config,
            items=items.children,
        )

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", self.config),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("temp_var_decls", comments=True)
class TemporaryVariableDeclarations(VariableDeclarationBlock):
    items: List[VariableInitDeclaration]

    @staticmethod
    def from_lark(items: lark.Tree) -> TemporaryVariableDeclarations:
        return TemporaryVariableDeclarations(items.children)

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_TEMP",
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("var_inst_declaration", comments=True)
class MethodInstanceVariableDeclarations(VariableDeclarationBlock):
    items: List[VariableInitDeclaration]

    @staticmethod
    def from_lark(items: lark.Tree) -> MethodInstanceVariableDeclarations:
        return MethodInstanceVariableDeclarations(items.children)

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_INST",
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("located_var_decl", comments=True)
class LocatedVariableDeclaration:
    name: Optional[lark.Token]
    location: Location
    init: LocatedVariableSpecInit

    @_commented_block
    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init}"


@dataclass
@_rule_handler("located_var_declarations", comments=True)
class LocatedVariableDeclarations(VariableDeclarationBlock):
    config: Optional[lark.Token]
    persistent: bool
    items: List[LocatedVariableDeclaration]

    @staticmethod
    def from_lark(
        config: Optional[lark.Token],
        persistent: Optional[lark.Token],
        *items: LocatedVariableDeclaration,
    ) -> LocatedVariableDeclarations:
        return LocatedVariableDeclarations(
            config=config,
            persistent=persistent is not None,
            items=list(items),
        )

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                join_if(
                    join_if("VAR", " ", self.config),
                    " ",
                    self.persistent and "PERSISTENT" or None
                ),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


IncompleteLocatedVariableSpecInit = Union[
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
    name: lark.Token
    location: IncompleteLocation
    init: IncompleteLocatedVariableSpecInit

    @_commented_block
    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init}"


@dataclass
@_rule_handler("incomplete_located_var_declarations", comments=True)
class IncompleteLocatedVariableDeclarations(VariableDeclarationBlock):
    retain: bool
    items: List[IncompleteLocatedVariableDeclaration]

    @staticmethod
    def from_lark(
        retain: Optional[lark.Token],
        *items: IncompleteLocatedVariableDeclaration,
    ) -> IncompleteLocatedVariableDeclarations:
        return IncompleteLocatedVariableDeclarations(
            retain=retain is not None,
            items=list(items),
        )

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", "RETAIN" if self.retain else None),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("external_declaration", comments=True)
class ExternalVariableDeclaration:
    name: lark.Token
    spec: Union[
        lark.Token,  # SIMPLE_SPECIFICATION / STRUCTURE_TYPE_NAME / FUNCTION_BLOCK_TYPE_NAME
        SubrangeSpecification,
        EnumeratedSpecification,
        ArraySpecification,
    ]

    @_commented_block
    def __str__(self) -> str:
        return f"{self.name} : {self.spec}"


@dataclass
@_rule_handler("external_var_declarations", comments=True)
class ExternalVariableDeclarations(VariableDeclarationBlock):
    constant: bool
    items: List[ExternalVariableDeclaration]

    @staticmethod
    def from_lark(
        constant: Optional[lark.Token],
        *items: ExternalVariableDeclaration,
    ) -> ExternalVariableDeclarations:
        return ExternalVariableDeclarations(
            constant=constant is not None,
            items=list(items),
        )

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_EXTERNAL", " ", self.constant and "CONSTANT" or None),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("input_declarations", comments=True)
class InputDeclarations(VariableDeclarationBlock):
    retain: Optional[lark.Token]
    items: List[InputDeclaration]

    @staticmethod
    def from_lark(retain: Optional[lark.Token], *items: InputDeclaration) -> InputDeclarations:
        return InputDeclarations(retain, list(items) if items else [])

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_INPUT", " ", self.retain),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("output_declarations", comments=True)
class OutputDeclarations(VariableDeclarationBlock):
    retain: Optional[lark.Token]
    items: List[OutputDeclaration]

    @staticmethod
    def from_lark(retain: Optional[lark.Token], items: lark.Tree) -> OutputDeclarations:
        return OutputDeclarations(retain, items.children)

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_OUTPUT", " ", self.retain),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("input_output_declarations", comments=True)
class InputOutputDeclarations(VariableDeclarationBlock):
    items: List[InputOutputDeclaration]

    @staticmethod
    def from_lark(items: lark.Tree) -> InputOutputDeclarations:
        return InputOutputDeclarations(items.children)

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_IN_OUT",
                *(indent_if(f"{item};") for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("program_access_decl", comments=True)
class AccessDeclaration:
    name: lark.Token
    variable: SymbolicVariable
    type_name: DataType
    direction: Optional[lark.Token]

    @_commented_block
    def __str__(self) -> str:
        return join_if(
            f"{self.name} : {self.variable} : {self.type_name}",
            " ",
            self.direction
        )


@dataclass
@_rule_handler("function_var_declarations", comments=True)
class FunctionVariableDeclarations(VariableDeclarationBlock):
    constant: Optional[lark.Token]
    items: List[VariableInitDeclaration]

    @staticmethod
    def from_lark(
        constant: Optional[lark.Token],
        body: lark.Tree,
    ) -> FunctionVariableDeclarations:
        return FunctionVariableDeclarations(
            constant=constant is not None,
            items=body.children,
        )

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                ("VAR CONSTANT" if self.constant else "VAR"),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("program_access_decls", comments=True)
class AccessDeclarations(VariableDeclarationBlock):
    items: List[AccessDeclaration]

    @staticmethod
    def from_lark(*items: AccessDeclaration) -> AccessDeclarations:
        return AccessDeclarations(list(items))

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_ACCESS",
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("global_var_declarations", comments=True)
class GlobalVariableDeclarations(VariableDeclarationBlock):
    constant: bool
    retain: bool
    persistent: bool
    items: List[GlobalVariableDeclaration]

    @staticmethod
    def from_lark(
        const_or_retain: Optional[lark.Token],
        persistent: Optional[lark.Token],
        *items: GlobalVariableDeclaration
    ) -> GlobalVariableDeclarations:
        return GlobalVariableDeclarations(
            constant=str(const_or_retain).lower() == "constant",
            retain=str(const_or_retain).lower() == "retain",
            persistent=persistent is not None,
            items=list(items)
        )

    @_commented_block
    def __str__(self) -> str:
        options = []
        if self.constant:
            options.append("CONSTANT")
        if self.retain:
            options.append("RETAIN")
        if self.persistent:
            options.append("PERSISTENT")
        return "\n".join(
            (
                join_if("VAR_GLOBAL", " ", " ".join(options) if options else None),
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


class Statement:
    ...


@dataclass
@_rule_handler("else_if_clause", comments=True)
class ElseIfClause:
    if_expression: Expression
    statements: Optional[StatementList]

    @_commented_block
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

    @_commented_block
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

    @staticmethod
    def from_lark(
        if_expr: Expression,
        then: Optional[StatementList],
        *args: Union[ElseIfClause, ElseClause]
    ) -> IfStatement:
        else_clause: Optional[ElseClause] = None
        if args and isinstance(args[-1], ElseClause) or args[-1] is None:
            else_clause = args[-1]
            args = args[:-1]

        else_ifs: List[ElseIfClause] = list(args)
        return IfStatement(
            if_expression=if_expr,
            statements=then,
            else_ifs=else_ifs,
            else_clause=else_clause,
        )

    @_commented_block
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


@dataclass
@_rule_handler("case_element", comments=True)
class CaseElement(Statement):
    matches: List[Union[Subrange, Integer, EnumeratedValue, SymbolicVariable]]
    statements: Optional[StatementList]

    @staticmethod
    def from_lark(
        matches: lark.Tree,
        statements: Optional[StatementList],
    ) -> CaseElement:
        return CaseElement(
            matches=matches.children,
            statements=statements,
        )

    @_commented_block
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
    cases: List[StatementList]
    else_clause: Optional[ElseClause]

    @staticmethod
    def from_lark(
        expr: Expression,
        *args: Union[CaseStatement, ElseClause]
    ) -> CaseStatement:
        else_clause = None
        if args and isinstance(args[-1], ElseClause) or args[-1] is None:
            else_clause = args[-1]
            args = args[:-1]
        return CaseStatement(
            expression=expr,
            cases=list(args),
            else_clause=else_clause,
        )

    @_commented_block
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
    variable: lark.Token

    @_commented_block
    def __str__(self):
        return f"{self.variable};"


@dataclass
@_rule_handler("action_statement", comments=True)
class ActionStatement(Statement):
    # TODO: overlaps with no-op statement?
    action: lark.Token

    @_commented_block
    def __str__(self):
        return f"{self.action};"


@dataclass
@_rule_handler("set_statement", comments=True)
class SetStatement(Statement):
    variable: SymbolicVariable
    expression: Expression

    @_commented_block
    def __str__(self):
        return f"{self.variable} S= {self.expression};"


@dataclass
@_rule_handler("reference_assignment_statement", comments=True)
class ReferenceAssignmentStatement(Statement):
    variable: SymbolicVariable
    expression: Expression

    @_commented_block
    def __str__(self):
        return f"{self.variable} REF= {self.expression};"


@dataclass
@_rule_handler("reset_statement")
class ResetStatement(Statement):
    variable: SymbolicVariable
    expression: Expression

    def __str__(self):
        return f"{self.variable} R= {self.expression};"


@dataclass
@_rule_handler("exit_statement", comments=True)
class ExitStatement(Statement):
    @_commented_block
    def __str__(self):
        return "EXIT;"


@dataclass
@_rule_handler("return_statement", comments=True)
class ReturnStatement(Statement):
    @_commented_block
    def __str__(self):
        return "RETURN;"


@dataclass
@_rule_handler("assignment_statement", comments=True)
class AssignmentStatement(Statement):
    variables: List[lark.Token]
    expression: Expression

    @staticmethod
    def from_lark(*args) -> AssignmentStatement:
        *variables, expression = args
        return AssignmentStatement(
            variables=list(variables),
            expression=expression
        )

    @_commented_block
    def __str__(self):
        variables = " := ".join(str(var) for var in self.variables)
        return f"{variables} := {self.expression};"


@dataclass
@_rule_handler("method_statement", comments=True)
class MethodStatement(Statement):
    method: SymbolicVariable

    @_commented_block
    def __str__(self):
        return f"{self.method}();"


@dataclass
@_rule_handler("while_statement", comments=True)
class WhileStatement(Statement):
    expression: Expression
    statements: StatementList

    @_commented_block
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

    @_commented_block
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
    control: lark.Token
    from_: Expression
    to: Expression
    step: Optional[Expression]
    statements: StatementList

    @_commented_block
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

    @staticmethod
    def from_lark(
        *statements: Statement
    ) -> StatementList:
        return StatementList(
            statements=list(statements)
        )

    def __str__(self) -> str:
        def stringify_statement(statement: Union[Statement, FunctionBlockInvocation]) -> str:
            # TODO: this is a bit of a bug; who has the responsibility to
            # append a semicolon?
            if not isinstance(statement, Statement):
                return f"{statement};"
            return str(statement)

        return "\n".join(
            stringify_statement(statement) for statement in self.statements
        )


class IL_Parameter:
    ...


@dataclass
@_rule_handler("il_param_operand_assignment")
class IL_ParameterOperandAssignment(IL_Parameter):
    variable: SymbolicVariable
    value: IL_ParameterOperand

    def __str__(self) -> str:
        return f"{self.variable} := {self.value}"


@dataclass
@_rule_handler("il_param_instruction_assignment")
class IL_ParameterInstructionAssignment(IL_Parameter):
    variable: SymbolicVariable
    instructions: List[IL_Instruction]

    @staticmethod
    def from_lark(
        variable: SymbolicVariable,
        *instructions: IL_Instruction,
    ) -> IL_ParameterInstructionAssignment:
        return IL_ParameterInstructionAssignment(
            variable=variable,
            instructions=list(instructions),
        )

    def __str__(self) -> str:
        instructions = "\n".join(
            str(instruction) for instruction in self.instructions
        )
        return "\n".join(
            (
                f"{self.variable} := (",
                indent_if(instructions),
                ")",
            )
        )


class IL_Operation:
    ...


@dataclass
@_rule_handler("il_simple_operation")
class IL_SimpleOperation(IL_Operation):
    function: lark.Token
    operands: List[IL_ParameterOperand]

    @staticmethod
    def from_lark(
        function_or_operator: lark.Token,
        _,
        *operands: IL_ParameterOperand,
    ) -> IL_SimpleOperation:
        return IL_SimpleOperation(
            function=function_or_operator,
            operands=list(operands),
        )

    def __str__(self) -> str:
        operands = ", ".join(str(operand) for operand in self.operands)
        return f"{self.function} {operands}"


@dataclass
@_rule_handler("il_return_operator")
class IL_Return(IL_Operation):
    return_token: lark.Token

    def __str__(self) -> str:
        return str(self.return_token).upper()


@dataclass
@_rule_handler("il_expression")
class IL_Expression(IL_Operation):
    operator: lark.Token
    operand: Optional[IL_ParameterOperand]
    instructions: List[IL_SimpleInstruction]

    @staticmethod
    def from_lark(
        operator: lark.Token,
        operand: Optional[IL_ParameterOperand],
        *instructions: IL_SimpleInstruction,
    ) -> IL_FunctionCall:
        return IL_Expression(
            operator=operator,
            operand=operand,
            instructions=list(instructions),
        )

    def __str__(self) -> str:
        instructions = "\n".join(
            str(instruction)
            for instruction in self.instructions
        )
        instructions = f"\n{indent_if(instructions)}\n" if instructions else ""
        operand = (str(self.operand) if self.operand else "")
        return f"{self.operator}({operand}{instructions})"


@dataclass
@_rule_handler("il_jump_operation")
class IL_JumpOperation(IL_Operation):
    operator: lark.Token
    label: lark.Token

    def __str__(self) -> str:
        return f"{self.operator.upper()} {self.label}"


@dataclass
@_rule_handler("il_formal_function_call")
class IL_FunctionCall(IL_Operation):
    function: lark.Token
    parameters: List[IL_Parameter]

    @staticmethod
    def from_lark(
        function: lark.Token,
        *parameters: IL_Parameter,
    ) -> IL_FunctionCall:
        return IL_FunctionCall(function=function, parameters=list(parameters))

    def __str__(self) -> str:
        parameters = "".join(
            f"{indent_if(param)}\n"
            for param in self.parameters
        )
        return f"{self.function}({parameters})"


@dataclass
@_rule_handler("il_param_out_assignment")
class IL_ParameterOutAssignment(IL_Parameter):
    inverted: bool
    variable: SymbolicVariable
    assign_to: Variable

    @staticmethod
    def from_lark(
        not_token: Optional[lark.Token],
        variable: SymbolicVariable,
        assign_to: Variable,
    ) -> IL_ParameterOutAssignment:
        return IL_ParameterOutAssignment(
            inverted=not_token is not None,
            variable=variable,
            assign_to=assign_to,
        )

    def __str__(self) -> str:
        not_prefix = "NOT " if self.inverted else ""
        return f"{not_prefix}{self.variable} => {self.assign_to}"


@dataclass
@_rule_handler("il_fb_call")
class IL_FunctionBlockCall:
    operator: lark.Token
    function_block: lark.Token
    arguments: Union[List[IL_Parameter], List[IL_ParameterOperand]]

    @staticmethod
    def from_lark(
        operator: lark.Token,
        function_block: lark.Token,
        *arguments: Union[IL_Parameter, IL_ParameterOperand],
    ) -> IL_FunctionCall:
        return IL_FunctionBlockCall(
            operator=operator,
            function_block=function_block,
            arguments=list(arguments)
        )

    def __str__(self) -> str:
        is_param_list = self.arguments and isinstance(self.arguments[0], IL_Parameter)
        if is_param_list:
            arguments = ",\n".join(
                indent_if(arg)
                for arg in self.arguments
            )
            arguments = f"\n{arguments}\n"
        else:
            arguments = ", ".join(str(arg) for arg in self.arguments)

        return f"{self.operator.upper()} {self.function_block}({arguments})"


@dataclass
@_rule_handler("il_instruction")
class IL_Instruction:
    label: Optional[lark.Token]
    operation: Optional[IL_Operation]

    def __str__(self) -> str:
        if not self.label and not self.operation:
            return ""
        label = f"{self.label}:" if self.label else None
        return join_if(label, " ", self.operation)


IL_SimpleInstruction = Union[
    IL_SimpleOperation,
    IL_Expression,
    IL_FunctionCall,
]

IL_ParameterOperand = Union[
    Literal,
    Variable,
    EnumeratedValue,
]


@dataclass
@_rule_handler("instruction_list")
class InstructionList:
    instructions: List[IL_Instruction]

    @staticmethod
    def from_lark(
        *instructions: IL_Instruction
    ) -> InstructionList:
        return InstructionList(
            instructions=list(instructions)
        )

    def __str__(self) -> str:
        return "\n".join(
            str(instruction)
            for instruction in self.instructions
        )


@dataclass
@_rule_handler("config_access_declarations", comments=True)
class ConfigAccessDeclarations(VariableDeclarationBlock):
    items: List[ConfigAccessDeclaration]

    @staticmethod
    def from_lark(*items: ConfigAccessDeclaration) -> ConfigAccessDeclarations:
        return ConfigAccessDeclarations(list(items))

    @_commented_block
    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_ACCESS",
                *(textwrap.indent(f"{item};", INDENT) for item in self.items),
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("config_access_declaration", comments=True)
class ConfigAccessDeclaration:
    name: lark.Token
    path: ConfigAccessPath
    type_name: DataType
    direction: Optional[lark.Token]

    @_commented_block
    def __str__(self) -> str:
        return join_if(
            f"{self.name} : {self.path} : {self.type_name}",
            " ",
            self.direction
        )


@dataclass
@_rule_handler("direct_access_path")
class DirectAccessPath:
    resource_name: Optional[lark.Token]
    variable: DirectVariable

    def __str__(self):
        return join_if(self.resource_name, ".", self.variable)


@dataclass
@_rule_handler("access_path")
class SymbolicAccessPath:
    # NOTE: these attributes are ambiguous; consider just making into a
    # "DOTTED_IDENTIFIER".
    resource_name: Optional[lark.Token]
    program_name: Optional[lark.Token]
    function_block_names: List[lark.Token]
    variable: SymbolicVariable

    @staticmethod
    def from_lark(
        resource_name: Optional[lark.Token],
        program_name: Optional[lark.Token],
        *remainder: Union[lark.Token, SymbolicVariable],
    ) -> SymbolicAccessPath:
        *fb_names, variable = remainder
        return SymbolicAccessPath(
            resource_name=resource_name,
            program_name=program_name,
            function_block_names=list(fb_names),
            variable=variable
        )

    def __str__(self):
        return ".".join(
            str(part) for part in [
                self.resource_name,
                self.program_name,
                *self.function_block_names,
                self.variable
            ]
            if part is not None
        )


ConfigAccessPath = Union[DirectAccessPath, SymbolicAccessPath]


@dataclass
@_rule_handler("task_initialization")
class TaskInitialization:
    single: Optional[TaskDataSource]
    interval: Optional[TaskDataSource]
    priority: lark.Token

    def __str__(self):
        parts = " ".join(
            part for part in (
                f"SINGLE := {self.single}," if self.single else None,
                f"INTERVAL := {self.interval}," if self.interval else None,
                f"PRIORITY := {self.priority}"
            )
            if part is not None
        )
        return f"({parts})"


@dataclass
@_rule_handler("task_configuration", comments=True)
class TaskConfiguration:
    name: lark.Token
    init: TaskInitialization

    @_commented_block
    def __str__(self):
        return f"TASK {self.name} {self.init}"


TaskDataSource = Union[
    Literal,
    Variable,
]


TypeDeclarationItem = Union[
    ArrayTypeDeclaration,
    StructureTypeDeclaration,
    StringTypeDeclaration,
    SimpleTypeDeclaration,
    SubrangeTypeDeclaration,
    EnumeratedTypeDeclaration,
]


@dataclass
@_rule_handler("data_type_declaration", comments=True)
class DataTypeDeclaration:
    items: List[TypeDeclarationItem]

    @staticmethod
    def from_lark(*args: TypeDeclarationItem) -> DataTypeDeclaration:
        return DataTypeDeclaration(list(args))

    @_commented_block
    def __str__(self) -> str:
        if not self.items:
            return "TYPE\nEND_TYPE"

        items = "\n".join(
            textwrap.indent(f"{item};", INDENT) for item in self.items
        )
        return "\n".join(
            (
                f"TYPE {items.lstrip()}",
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
    # ConfigurationDeclaration,  # TODO
    GlobalVariableDeclarations,
]


@dataclass
@_rule_handler("iec_source")
class SourceCode:
    """Top-level source code item."""
    items: List[SourceCodeItem]

    @staticmethod
    def from_lark(*args: SourceCodeItem) -> SourceCode:
        return SourceCode(list(args))

    def __str__(self):
        return "\n".join(str(item) for item in self.items)


def _has_meta_kwarg(func: Callable) -> bool:
    sig = inspect.signature(func)
    try:
        meta_param = sig.parameters["meta"]
    except KeyError:
        return False

    return inspect.Parameter.KEYWORD_ONLY == meta_param.kind


def _annotator_wrapper(handler):
    def wrapped(self: GrammarTransformer, data: Any, children: list, meta: lark.tree.Meta) -> Any:
        result = handler(*children)
        if not isinstance(result, (lark.Tree, lark.Token, list)):
            result.meta = meta
        return result

    return wrapped


def pass_through(obj: Optional[T] = None) -> Optional[T]:
    """Transformer helper to pass through an optional single argument."""
    return obj


class GrammarTransformer(lark.visitors.Transformer):
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
    _filename: Optional[str]
    comments: List[lark.Token]

    def __init__(self, comments: Optional[List[lark.Token]] = None, fn=None):
        super().__init__()
        self._filename = fn
        self.comments = comments or []

    constant = _annotator_wrapper(pass_through)

    locals().update(
        **{
            name: _annotator_wrapper(handler)
            for name, handler in _class_handlers.items()
        }
    )

    def transform(self, tree):
        transformed = super().transform(tree)
        if self.comments:
            merge_comments(transformed, self.comments)
        return transformed

    @_annotator_wrapper
    def full_subrange():
        return FullSubrange()

    @_annotator_wrapper
    def var1_list(*items: DeclaredVariable) -> List[DeclaredVariable]:
        return list(items)

    @_annotator_wrapper
    def fb_decl_name_list(*items: lark.Token) -> List[lark.Token]:
        return list(items)

    @_annotator_wrapper
    def signed_integer(value: lark.Token):
        return Integer.from_lark(None, value)

    @_annotator_wrapper
    def integer(value: lark.Token):
        return Integer.from_lark(None, value)

    @_annotator_wrapper
    def true(value: lark.Token):
        return Boolean(value=value)

    @_annotator_wrapper
    def false(value: lark.Token):
        return Boolean(value=value)

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
