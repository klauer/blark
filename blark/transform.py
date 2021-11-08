from __future__ import annotations

import enum
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union

import lark

_rule_to_handler = {}


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


def _rule_handler(
    *rules: Union[str, List[str]]
) -> Callable[[Type[T]], Type[T]]:
    """Decorator - the wrapped class will handle the provided rules."""
    def wrapper(cls: Type[T]) -> Type[T]:
        for rule in rules:
            handler = _rule_to_handler.get(rule, None)
            if handler is not None:
                raise ValueError(f"Handler already specified for: {rule} ({handler})")

            _rule_to_handler[rule] = cls

        cls._lark_ = rules
        return cls

    return wrapper


@dataclass
class Literal:
    """Literal value."""

    def __str__(self) -> str:
        return str(self.value)


Constant = Literal  # an alias for now


@dataclass
@_rule_handler("integer_literal")
class Integer(Literal):
    """Integer literal value."""

    value: lark.Token
    type: Optional[lark.Token] = None
    base: int = 10

    @staticmethod
    def from_lark(
        type_name: Optional[lark.Token],
        value: Union[Integer, lark.Token],
        *,
        base: int = 10,
    ) -> Integer:
        if isinstance(value, Integer):
            # Adding type information; wrap Integer
            return Integer(
                type=type_name,
                value=value.value,
                base=value.base,
            )
        return Integer(
            type=type_name,
            value=value,
            base=base,
        )

    def __str__(self) -> str:
        value = f"{self.base}#{self.value}" if self.base != 10 else str(self.value)
        if self.type:
            return f"{self.type}#{value}"
        return value


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

    value: lark.Token
    type: Optional[lark.Token] = None
    base: int = 10

    @staticmethod
    def from_lark(
        type_name: Optional[lark.Token], value: lark.Token, *, base: int = 10
    ) -> BitString:
        return BitString(type=type_name, value=value, base=base)

    def __str__(self) -> str:
        value = f"{self.base}#{self.value}" if self.base != 10 else str(self.value)
        if self.type:
            return f"{self.type}#{value}"
        return value


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

    @staticmethod
    def from_lark(
        _: lark.Token, hour: lark.Tree, minute: lark.Tree, second: lark.Tree
    ) -> TimeOfDay:
        (hour,) = hour.children
        (minute,) = minute.children
        (second,) = second.children
        return TimeOfDay(
            hour=hour,
            minute=minute,
            second=second,
        )

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

    @staticmethod
    def from_lark(year: lark.Tree, month: lark.Tree, day: lark.Tree) -> Date:
        (year,) = year.children
        (month,) = month.children
        (day,) = day.children
        return Date(year=year, month=month, day=day)

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
            date=Date(
                year=year.children[0],
                month=month.children[0],
                day=day.children[0],
            ),
            time=TimeOfDay(
                hour=hour.children[0],
                minute=minute.children[0],
                second=second.children[0],
            ),
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


@_rule_handler("indirection_type")
@_rule_handler("pointer_type")
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
class DirectVariable(Expression):
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
class SymbolicVariable(Expression):
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
    type_name: Optional[lark.Token]
    value: Optional[Expression]

    def __str__(self) -> str:
        type_ = join_if(self.indirection, " ", self.type_name)
        return join_if(type_, " := ", self.value)


@dataclass
@_rule_handler("simple_type_declaration")
class TypeDeclaration:
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
@_rule_handler(
    "single_byte_string_spec",
    "double_byte_string_spec",
)
class StringTypeInitialization:
    string_type: lark.Token
    length: Optional[lark.Token]
    value: Optional[lark.Token]

    @staticmethod
    def from_lark(
        *args: lark.Token,
    ) -> StringTypeInitialization:
        if len(args) == 4:
            string_type, length, _, value = args
            return StringTypeInitialization(
                string_type, length, value
            )
        string_type, length = args
        return StringTypeInitialization(
            string_type, length, None
        )

    def __str__(self) -> str:
        type_and_length = join_if(self.string_type, "", self.length)
        return join_if(type_and_length, " := ", self.value)


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
    spec: Optional[lark.Token] = None
    value: Optional[Expression] = None

    def __str__(self) -> str:
        if self.indirection:
            spec = f"{self.indirection} {self.spec}"
        else:
            spec = f"{self.spec}"

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
    value: Optional[Integer]

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
@_rule_handler("enumerated_type_declaration")
class EnumeratedTypeDeclaration:
    name: lark.Token
    init: EnumeratedTypeInitialization

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
    def from_lark(*elements):
        return ArrayInitialization(elements=elements)

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
@_rule_handler("array_type_declaration")
class ArrayTypeDeclaration:
    name: lark.Token
    init: ArrayTypeInitialization

    def __str__(self) -> str:
        return f"{self.name} : {self.init}"


@dataclass
@_rule_handler("structure_type_declaration")
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

    def __str__(self) -> str:
        if self.declarations:
            indent = f"\n{INDENT}"
            declarations = indent + indent.join(str(decl) for decl in self.declarations)
        else:
            declarations = ""

        definition = join_if(self.name, " ", self.extends)
        indirection = f" {self.indirection}" if self.indirection else ""
        return "\n".join(
            (
                f"{definition} :{indirection}",
                f"STRUCT{declarations}",
                "END_STRUCT"
            )
        )


@dataclass
@_rule_handler("structure_element_declaration")
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
@_rule_handler("initialized_structure_type_declaration")
class InitializedStructureTypeDeclaration:
    name: lark.Token
    extends: Optional[lark.Token]
    init: StructureInitialization

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
        return f"{self.op} {self.expr}"


@dataclass
@_rule_handler(
    "expression",
    "add_expression",
    "and_expression",
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

        def get_operator_and_expr() -> Tuple[lark.Token, Expression]:
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
@_rule_handler("var1")
class VariableOne:
    name: lark.Token
    location: Optional[Union[IncompleteLocation, Location]]

    def __str__(self) -> str:
        return join_if(self.name, " ", self.location)


@dataclass
@_rule_handler("var1_list")
class VariableList:
    variables: List[VariableOne]

    @staticmethod
    def from_lark(*items) -> VariableList:
        return VariableList(list(items))

    def __str__(self) -> str:
        return ", ".join(str(variable) for variable in self.variables)


@dataclass
@_rule_handler("var1_init_decl")
class VariableOneInitDeclaration:
    variables: VariableList
    init: Union[TypeInitialization, SubrangeTypeInitialization, EnumeratedTypeInitialization]

    def __str__(self) -> str:
        return f"{self.variables} : {self.init}"


@dataclass
@_rule_handler("array_var_init_decl")
class ArrayVariableInitDeclaration:
    variables: VariableList
    init: ArrayTypeInitialization

    def __str__(self) -> str:
        return f"{self.variables} : {self.init}"


@dataclass
@_rule_handler("structured_var_init_decl")
class StructuredVariableInitDeclaration:
    variables: VariableList
    init: InitializedStructure

    def __str__(self) -> str:
        return f"{self.variables} : {self.init}"


@dataclass
@_rule_handler(
    "single_byte_string_var_declaration",
    "double_byte_string_var_declaration"
)
class StringVariableInitDeclaration:
    variables: VariableList
    type_name: lark.Token
    length: Optional[lark.Token]
    value: lark.Token

    @staticmethod
    def from_lark(variables: VariableList, string_info: StringTypeInitialization):
        return StringVariableInitDeclaration(
            variables=variables,
            type_name=string_info.string_type,
            length=string_info.length,
            value=string_info.value,
        )

    def __str__(self) -> str:
        type_name = join_if(self.type_name, "", self.length)
        return f"{self.variables} : {type_name} := {self.value}"


@dataclass
@_rule_handler("fb_decl_name_list")
class FunctionBlockDeclarationNameList:
    names: List[lark.Token]

    @staticmethod
    def from_lark(*names: lark.Token) -> FunctionBlockDeclarationNameList:
        return FunctionBlockDeclarationNameList(list(names))

    def __str__(self) -> str:
        return ", ".join(str(name) for name in self.names)


class FunctionBlockDeclaration:
    ...


@dataclass
@_rule_handler("fb_name_decl")
class FunctionBlockNameDeclaration(FunctionBlockDeclaration):
    names: FunctionBlockDeclarationNameList
    type_name: lark.Token
    init: Optional[StructureInitialization] = None

    def __str__(self) -> str:
        name_and_type = f"{self.names} : {self.type_name}"
        return join_if(name_and_type, " := ", self.init)


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
class FunctionBlockInvocation(FunctionBlockDeclaration):
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
@_rule_handler("fb_invocation_decl")
class FunctionBlockInvocationDeclaration(FunctionBlockDeclaration):
    names: FunctionBlockDeclarationNameList
    invocation: FunctionBlockInvocation

    def __str__(self) -> str:
        return f"{self.names} : {self.invocation}"


@dataclass
@_rule_handler("edge_declaration")
class EdgeDeclaration:
    variables: VariableList
    edge: lark.Token

    def __str__(self) -> str:
        return f"{self.variables} : BOOL {self.edge}"


@dataclass
@_rule_handler("global_var_spec")
class GlobalVariableSpec:
    names: List[lark.Token]
    location: Optional[Union[Location, IncompleteLocation]]

    @staticmethod
    def from_lark(
        name_or_names: Union[lark.Token, lark.Tree],
        location: Optional[Union[Location, IncompleteLocation]] = None
    ) -> GlobalVariableSpec:
        if location is None:
            return GlobalVariableSpec(
                names=name_or_names.children,
                location=None
            )

        return GlobalVariableSpec(
            names=[name_or_names],
            location=location
        )

    def __str__(self) -> str:
        if not self.location:
            return ", ".join(self.names)
        return f"{self.names[0]} : {self.location}"


LocatedVariableSpecInit = Union[
    TypeInitialization,
    SubrangeTypeInitialization,
    EnumeratedTypeInitialization,
    ArrayTypeInitialization,
    InitializedStructure,
    StringTypeInitialization,
]


@dataclass
@_rule_handler("global_var_decl")
class GlobalVariableDeclaration:
    spec: GlobalVariableSpec
    init: Union[
        LocatedVariableSpecInit,
        FunctionBlockInvocation,
        lark.Token  # FB type name
    ]

    def __str__(self) -> str:
        return f"{self.spec} : {self.init}"


@dataclass
@_rule_handler("extends")
class Extends:
    name: lark.Token

    def __str__(self) -> str:
        return f"EXTENDS {self.name}"


@dataclass
@_rule_handler("function_block_body")
class FunctionBlockBody:
    source: Union[
        StatementList,
        # SfcNetwork
    ]

    def __str__(self) -> str:
        return str(self.source)


@dataclass
@_rule_handler("function_block_type_declaration")
class FunctionBlock:
    name: lark.Token
    extends: Optional[lark.Token]
    declarations: List[VariableDeclarationBlock]
    body: Optional[FunctionBlockBody]

    @staticmethod
    def from_lark(
        derived_name: lark.Token,
        extends: Extends,
        *args
    ):
        *declarations, body = args
        return FunctionBlock(
            name=derived_name,
            extends=extends,
            declarations=list(declarations),
            body=body,
        )

    def __str__(self) -> str:
        return "\n".join(
            line for line in
            (
                join_if(f"FUNCTION_BLOCK {self.name}", " ", self.extends),
                *[str(declaration) for declaration in self.declarations],
                textwrap.indent(str(self.body), INDENT) if self.body else None,
                "END_FUNCTION_BLOCK",
            )
            if line is not None
        )


@dataclass
class VariableDeclarationBlock:
    ...


VarInitDeclaration = Union[
    ArrayVariableInitDeclaration,
    StringVariableInitDeclaration,
    VariableOneInitDeclaration,
    FunctionBlockDeclaration,
    # EdgeDeclaration,
]

InputOutputDeclaration = VarInitDeclaration
OutputDeclaration = VarInitDeclaration

InputDeclaration = Union[
    VarInitDeclaration,
    EdgeDeclaration,
]
GlobalVariableDeclarationType = Union[
    VarInitDeclaration,
    GlobalVariableDeclaration,
]
# FunctionBlockDeclarations = Union[
#     ...
# ]


@dataclass
@_rule_handler("var_declarations")
class VariableDeclarations(VariableDeclarationBlock):
    config: Optional[lark.Token]
    items: List[VarInitDeclaration]

    @staticmethod
    def from_lark(config: Optional[lark.Token], items: lark.Tree) -> VariableDeclarations:
        return VariableDeclarations(
            config=config,
            items=items.children,
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR", " ", self.config),
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("temp_var_decls")
class TemporaryVariableDeclarations(VariableDeclarationBlock):
    items: List[VarInitDeclaration]

    @staticmethod
    def from_lark(items: lark.Tree) -> TemporaryVariableDeclarations:
        return TemporaryVariableDeclarations(items.children)

    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_TEMP",
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("located_var_decl")
class LocatedVariableDeclaration:
    name: Optional[lark.Token]
    location: Location
    init: LocatedVariableSpecInit

    def __str__(self) -> str:
        name_and_location = join_if(self.name, " ", self.location)
        return f"{name_and_location} : {self.init}"


@dataclass
@_rule_handler("located_var_declarations")
class LocatedVariableDeclarations(VariableDeclarationBlock):
    config: Optional[lark.Token]
    persistent: bool
    items: List[LocatedVariableDeclaration]

    @staticmethod
    def from_lark(
        config: Optional[lark.Token],
        persistent: Optional[lark.Token],
        *items: LocatedVariableDeclaration,
    ) -> VariableDeclarations:
        return LocatedVariableDeclarations(
            config=config,
            persistent=persistent is not None,
            items=list(items),
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if(
                    join_if("VAR", " ", self.config),
                    " ",
                    self.persistent and "PERSISTENT" or None
                ),
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("external_declaration")
class ExternalVariableDeclaration:
    name: lark.Token
    spec: Union[
        lark.Token,  # SIMPLE_SPECIFICATION / STRUCTURE_TYPE_NAME / FUNCTION_BLOCK_TYPE_NAME
        SubrangeSpecification,
        EnumeratedSpecification,
        ArraySpecification,
    ]

    def __str__(self) -> str:
        return f"{self.name} : {self.spec}"


@dataclass
@_rule_handler("external_var_declarations")
class ExternalVariableDeclarations(VariableDeclarationBlock):
    constant: bool
    items: List[ExternalVariableDeclaration]

    @staticmethod
    def from_lark(
        constant: Optional[lark.Token],
        *items: ExternalVariableDeclaration,
    ) -> VariableDeclarations:
        return ExternalVariableDeclarations(
            constant=constant is not None,
            items=list(items),
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_EXTERNAL", " ", self.constant and "CONSTANT" or None),
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("input_declarations")
class InputDeclarations(VariableDeclarationBlock):
    retain: Optional[lark.Token]
    items: List[InputDeclaration]

    @staticmethod
    def from_lark(retain: Optional[lark.Token], *items: InputDeclaration) -> InputDeclarations:
        return InputDeclarations(retain, list(items) if items else [])

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_INPUT", " ", self.retain),
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("output_declarations")
class OutputDeclarations(VariableDeclarationBlock):
    retain: Optional[lark.Token]
    items: List[OutputDeclaration]

    @staticmethod
    def from_lark(retain: Optional[lark.Token], items: lark.Tree) -> OutputDeclarations:
        return OutputDeclarations(retain, items.children)

    def __str__(self) -> str:
        return "\n".join(
            (
                join_if("VAR_OUTPUT", " ", self.retain),
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("input_output_declarations")
class InputOutputDeclarations(VariableDeclarationBlock):
    items: List[InputOutputDeclaration]

    @staticmethod
    def from_lark(items: lark.Tree) -> InputOutputDeclarations:
        return InputOutputDeclarations(items.children)

    def __str__(self) -> str:
        return "\n".join(
            (
                "VAR_IN_OUT",
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


@dataclass
@_rule_handler("global_var_declarations")
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
                *[f"{INDENT}{item};" for item in self.items],
                "END_VAR",
            )
        )


class Statement:
    ...


@dataclass
@_rule_handler("else_if_clause")
class ElseIfClause:
    if_expression: Expression
    statements: Optional[StatementList]

    def __str__(self):
        return "\n".join(
            s for s in (
                f"ELSIF {self.if_expression} THEN",
                indent_if(self.statements),
            )
            if s is not None
        )


@dataclass
@_rule_handler("else_clause")
class ElseClause:
    statements: Optional[StatementList]

    def __str__(self):
        return "\n".join(
            s for s in (
                "ELSE",
                indent_if(self.statements),
            )
            if s is not None
        )


@dataclass
@_rule_handler("if_statement")
class IfStatement(Statement):
    if_expression: Expression
    statements: Optional[StatementList]
    else_ifs: Optional[List[ElseIfClause]]
    else_clause: Optional[ElseClause]

    @staticmethod
    def from_lark(
        if_expr: Expression,
        then: Optional[StatementList],
        *args: Union[ElseIfClause, ElseClause]
    ) -> IfStatement:
        else_clause = None
        if args and isinstance(args[-1], ElseClause) or args[-1] is None:
            else_clause = args[-1]
            args = args[:-1]
        return IfStatement(
            if_expression=if_expr,
            statements=then,
            else_ifs=list(args),
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


@dataclass
@_rule_handler("case_element")
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
@_rule_handler("case_statement")
class CaseStatement(Statement):
    expression: Expression
    cases: Optional[StatementList]
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

    def __str__(self):
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
@_rule_handler("no_op_statement")
class NoOpStatement(Statement):
    variable: lark.Token

    def __str__(self):
        return f"{self.variable};"


@dataclass
@_rule_handler("action_statement")
class ActionStatement(Statement):
    # TODO: overlaps with no-op statement?
    action: lark.Token

    def __str__(self):
        return f"{self.action};"


@dataclass
@_rule_handler("set_statement")
class SetStatement(Statement):
    variable: SymbolicVariable
    expression: Expression

    def __str__(self):
        return f"{self.variable} S= {self.expression};"


@dataclass
@_rule_handler("reference_assignment_statement")
class ReferenceAssignmentStatement(Statement):
    variable: SymbolicVariable
    expression: Expression

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
@_rule_handler("exit_statement")
class ExitStatement(Statement):
    def __str__(self):
        return "EXIT;"


@dataclass
@_rule_handler("return_statement")
class ReturnStatement(Statement):
    def __str__(self):
        return "RETURN;"


@dataclass
@_rule_handler("assignment_statement")
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

    def __str__(self):
        variables = " := ".join(str(var) for var in self.variables)
        return f"{variables} := {self.expression};"


@dataclass
@_rule_handler("method_statement")
class MethodStatement(Statement):
    method: SymbolicVariable

    def __str__(self):
        return f"{self.method}();"


@dataclass
@_rule_handler("while_statement")
class WhileStatement(Statement):
    expression: Expression
    statements: StatementList

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
@_rule_handler("repeat_statement")
class RepeatStatement(Statement):
    statements: StatementList
    expression: Expression

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


@dataclass
class Body:
    ...


@staticmethod
def pass_through(obj: Optional[T] = None) -> Optional[T]:
    """Transformer helper to pass through an optional single argument."""
    return obj


@lark.visitors.v_args(inline=True)
class GrammarTransformer(lark.visitors.Transformer):
    def __init__(self, fn=None):
        super().__init__()
        self._filename = fn
        self.__dict__.update(**_class_handlers)

    constant = pass_through

    def full_subrange(self):
        return FullSubrange()

    def integer(self, value: lark.Token):
        return Integer.from_lark(None, value)

    def binary_integer(self, value: lark.Token):
        return Integer.from_lark(None, value, base=2)

    def octal_integer(self, value: lark.Token):
        return Integer.from_lark(None, value, base=8)

    def hex_integer(self, value: lark.Token):
        return Integer.from_lark(None, value, base=16)

    def binary_bit_string_literal(self, type_name: lark.Token, value: lark.Token):
        return BitString.from_lark(type_name, value, base=2)

    def octal_bit_string_literal(self, type_name: lark.Token, value: lark.Token):
        return BitString.from_lark(type_name, value, base=8)

    def hex_bit_string_literal(self, type_name: lark.Token, value: lark.Token):
        return BitString.from_lark(type_name, value, base=16)

    def true(self, value: lark.Token):
        return Boolean(value=value)

    def false(self, value: lark.Token):
        return Boolean(value=value)


def _get_default_instantiator(cls: type):
    def instantiator(*args):
        return cls(*args)

    return instantiator


def _get_class_handlers():
    result = {}
    for cls in globals().values():
        if hasattr(cls, "_lark_"):
            token_names = cls._lark_
            if isinstance(token_names, str):
                token_names = [token_names]
            for token_name in token_names:
                if token_name in result:
                    raise ValueError(f"Saw {token_name!r} twice")
                if not hasattr(cls, "from_lark"):
                    cls.from_lark = _get_default_instantiator(cls)
                result[token_name] = lark.visitors.v_args(inline=True)(cls.from_lark)

    return result


_class_handlers = _get_class_handlers()
