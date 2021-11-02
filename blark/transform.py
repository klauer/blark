from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Type, TypeVar, Union

import lark

_rule_to_handler = {}


T = TypeVar("T")


def _rule_handler(rules: Union[str, List[str]]) -> Callable[[Type[T]], Type[T]]:
    """Decorator - the wrapped class will handle the provided rules."""
    if isinstance(rules, str):
        rules = [rules]

    def wrapper(cls: Type[T]) -> Type[T]:
        for rule in rules:
            handler = _rule_to_handler.get(rule, None)
            if handler is not None:
                raise ValueError(
                    f"Handler already specified for: {rule} ({handler})"
                )

            _rule_to_handler[rule] = cls

        if hasattr(cls, "_lark_"):
            cls._lark_ += rules
        else:
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
    def from_lark(type_name: Optional[lark.Token], value: Union[Integer, lark.Token], *, base: int = 10) -> Integer:
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
        value = (
            f"{self.base}#{self.value}"
            if self.base != 10
            else str(self.value)
        )
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
    def from_lark(type_name: Optional[lark.Token], value: lark.Token, *, base: int = 10) -> BitString:
        return BitString(type=type_name, value=value, base=base)

    def __str__(self) -> str:
        value = (
            f"{self.base}#{self.value}"
            if self.base != 10
            else str(self.value)
        )
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
        kwargs = {
            tree.data: tree.children[0]
            for tree in interval.iter_subtrees()
        }

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
    def from_lark(_: lark.Token, hour: lark.Tree, minute: lark.Tree, second: lark.Tree) -> TimeOfDay:
        hour, = hour.children
        minute, = minute.children
        second, = second.children
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
        year, = year.children
        month, = month.children
        day, = day.children
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
    def from_lark(year: lark.Token, month: lark.Token, day: lark.Token, hour: lark.Token, minute: lark.Token, second: lark.Token) -> DateTime:
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
class Extends:
    name: lark.Token


@dataclass
class FunctionBlock:
    name: lark.Token
    extends: Optional[lark.Token]
    declarations: tuple[VariableDeclarationBlock, ...]
    body: Optional[Code]

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


@lark.visitors.v_args(inline=True)
class GrammarTransformer(lark.visitors.Transformer):
    def __init__(self, fn=None):
        super().__init__()
        self._filename = fn
        self.__dict__.update(**_class_handlers)

    def extends(self, name: lark.Token):
        return Extends(name=name)

    def function_block_body(self, *items):
        return Body()

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


def _get_class_handlers():
    result = {}
    for obj in globals().values():
        if hasattr(obj, "_lark_"):
            token_names = obj._lark_
            if isinstance(token_names, str):
                token_names = [token_names]
            for token_name in token_names:
                result[token_name] = lark.visitors.v_args(inline=True)(obj.from_lark)

    return result

_class_handlers = _get_class_handlers()
