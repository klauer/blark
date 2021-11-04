from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple, Type, TypeVar, Union

import lark

_rule_to_handler = {}


T = TypeVar("T")


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
class Expression:
    ...


@dataclass
class Variable(Expression):
    ...


@_rule_handler("indirection_type")
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

    @staticmethod
    def from_lark(
        indirection: Optional[IndirectionType],
        type_name: lark.Token,
        value: Expression
    ) -> TypeInitialization:
        return TypeInitialization(
            indirection=indirection,
            type_name=type_name,
            value=value,
        )

    def __str__(self) -> str:
        if self.indirection:
            type_ = f"{self.indirection} {self.type_name}"
        else:
            type_ = f"{self.type_name}"

        if not self.value:
            return type_

        return f"{type_} := {self.value}"


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
@_rule_handler("extends")
class Extends:
    name: lark.Token

    def __str__(self) -> str:
        return f"EXTENDS {self.name}"


@dataclass
class FunctionBlock:
    name: lark.Token
    extends: Optional[lark.Token]
    declarations: tuple[VariableDeclarationBlock, ...]
    body: Optional[...]

    @staticmethod
    def from_lark(derived_name, extends, *declarations, body):
        return FunctionBlock(
            name=derived_name,
            extends=extends,
            declarations=declarations,
            body=body,
        )


@dataclass
class VariableDeclarationBlock:
    ...


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

    # def function_block_body(self, *items):
    #     return Body()

    def full_subrange(self):
        return FullSubrange()

    def fb_var_declaration(self, *items):
        return VariableDeclarationBlock()

    def function_block_declaration(self, *items):
        return items

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
                if not hasattr(cls, "from_lark"):
                    cls.from_lark = _get_default_instantiator(cls)
                result[token_name] = lark.visitors.v_args(inline=True)(cls.from_lark)

    return result


_class_handlers = _get_class_handlers()
