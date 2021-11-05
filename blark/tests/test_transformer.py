import pathlib

import pytest

from .. import transform as tf
from .conftest import get_grammar, stringify_tokens

TEST_PATH = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    "name, value, expected",
    [
        pytest.param("integer_literal", "-12", tf.Integer(value="-12")),
        pytest.param("integer_literal", "12", tf.Integer(value="12")),
        pytest.param("integer_literal", "INT#12", tf.Integer(value="12", type="INT")),
        pytest.param("integer_literal", "2#10010", tf.Integer(value="10010", base=2)),
        pytest.param("integer_literal", "8#22", tf.Integer(value="22", base=8)),
        pytest.param("integer_literal", "16#12", tf.Integer(value="12", base=16)),
        pytest.param("integer_literal", "UDINT#12", tf.Integer(value="12", base=10, type="UDINT")),
        pytest.param("integer_literal", "UDINT#2#010", tf.Integer(value="010", base=2, type="UDINT")),  # noqa: E501
        pytest.param("integer_literal", "UDINT#2#1001_0011", tf.Integer(value="1001_0011", base=2, type="UDINT")),  # noqa: E501
        pytest.param("integer_literal", "DINT#16#C0FFEE", tf.Integer(value="C0FFEE", base=16, type="DINT")),  # noqa: E501
        pytest.param("real_literal", "-12.0", tf.Real(value="-12.0")),
        pytest.param("real_literal", "12.0", tf.Real(value="12.0")),
        pytest.param("real_literal", "12.0e5", tf.Real(value="12.0e5")),
        pytest.param("bit_string_literal", "WORD#1234", tf.BitString(type="WORD", value="1234")),
        pytest.param("bit_string_literal", "WORD#2#0101", tf.BitString(type="WORD", value="0101", base=2)),  # noqa: E501
        pytest.param("bit_string_literal", "WORD#8#777", tf.BitString(type="WORD", value="777", base=8)),  # noqa: E501
        pytest.param("bit_string_literal", "word#16#FEEE", tf.BitString(type="word", value="FEEE", base=16)),  # noqa: E501
        pytest.param("boolean_literal", "BOOL#1", tf.Boolean(value="1")),
        pytest.param("boolean_literal", "BOOL#0", tf.Boolean(value="0")),
        pytest.param("boolean_literal", "BOOL#TRUE", tf.Boolean(value="TRUE")),
        pytest.param("boolean_literal", "BOOL#FALSE", tf.Boolean(value="FALSE")),
        pytest.param("duration", "TIME#1D", tf.Duration(days="1")),
        pytest.param("duration", "TIME#10S", tf.Duration(seconds="10")),
        pytest.param("duration", "TIME#1H", tf.Duration(hours="1")),
        pytest.param("duration", "TIME#1M", tf.Duration(minutes="1")),
        pytest.param("duration", "TIME#10MS", tf.Duration(milliseconds="10")),
        pytest.param("duration", "TIME#1.1D", tf.Duration(days="1.1")),
        pytest.param("duration", "TIME#10.1S", tf.Duration(seconds="10.1")),
        pytest.param("duration", "TIME#1.1H", tf.Duration(hours="1.1")),
        pytest.param("duration", "TIME#1.1M", tf.Duration(minutes="1.1")),
        pytest.param("duration", "TIME#10.1MS", tf.Duration(milliseconds="10.1")),
        pytest.param("duration", "T#1D1H1M1S1MS", tf.Duration(days="1", hours="1", minutes="1", seconds="1", milliseconds="1")),  # noqa: E501
        pytest.param("duration", "TIME#1H1M1S1MS", tf.Duration(hours="1", minutes="1", seconds="1", milliseconds="1")),  # noqa: E501
        pytest.param("time_of_day", "TIME_OF_DAY#1:1:1.2", tf.TimeOfDay(hour="1", minute="1", second="1.2")),  # noqa: E501
        # pytest.param("single_byte_character_string", "'abc'"),
        # pytest.param("double_byte_character_string", '"abc"'),
        # pytest.param("single_byte_string_spec", "STRING[1]"),
        # pytest.param("single_byte_string_spec", "STRING(1)"),
        # pytest.param("single_byte_string_spec", "STRING(1) := 'abc'"),
        # pytest.param("double_byte_string_spec", "WSTRING[1]"),
        # pytest.param("double_byte_string_spec", "WSTRING(1)"),
        # pytest.param("double_byte_string_spec", 'WSTRING(1) := "abc"'),
    ],
)
def test_literal(name, value, expected):
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    print(f"transformed -> {transformed}")
    assert stringify_tokens(transformed) == expected


@pytest.mark.parametrize(
    "name, value",
    [
        pytest.param("integer_literal", "-12"),
        pytest.param("integer_literal", "12"),
        pytest.param("integer_literal", "INT#12"),
        pytest.param("integer_literal", "2#10010"),
        pytest.param("integer_literal", "8#22"),
        pytest.param("integer_literal", "16#12"),
        pytest.param("integer_literal", "UDINT#12"),
        pytest.param("integer_literal", "UDINT#2#010"),
        pytest.param("integer_literal", "UDINT#2#1001_0011"),
        pytest.param("integer_literal", "DINT#16#C0FFEE"),
        pytest.param("real_literal", "-12.0"),
        pytest.param("real_literal", "12.0"),
        pytest.param("real_literal", "12.0e5"),
        pytest.param("bit_string_literal", "WORD#1234"),
        pytest.param("bit_string_literal", "WORD#2#0101"),
        pytest.param("bit_string_literal", "WORD#8#777"),
        pytest.param("bit_string_literal", "word#16#FEEE"),
        pytest.param("duration", "TIME#1D"),
        pytest.param("duration", "TIME#10S"),
        pytest.param("duration", "TIME#1H"),
        pytest.param("duration", "TIME#1M"),
        pytest.param("duration", "TIME#10MS"),
        pytest.param("duration", "TIME#1.1D"),
        pytest.param("duration", "TIME#10.1S"),
        pytest.param("duration", "TIME#1.1H"),
        pytest.param("duration", "TIME#1.1M"),
        pytest.param("duration", "TIME#10.1MS"),
        pytest.param("duration", "TIME#1D1H1M1S1MS"),
        pytest.param("duration", "TIME#1H1M1S1MS"),
        pytest.param("time_of_day", "TIME_OF_DAY#1:1:1.2"),
        pytest.param("date", "DATE#1970-1-1"),
        pytest.param("date_and_time", "DT#1970-1-1-1:2:30.3"),
    ],
)
def test_literal_roundtrip(name, value):
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    assert str(transformed) == value


@pytest.mark.parametrize(
    "name, value, expected",
    [
        pytest.param("boolean_literal", "BOOL#1", "TRUE"),
        pytest.param("boolean_literal", "BOOL#0", "FALSE"),
        pytest.param("boolean_literal", "BOOL#TRUE", "TRUE"),
        pytest.param("boolean_literal", "BOOL#FALSE", "FALSE"),
        pytest.param("boolean_literal", "1", "TRUE"),
        pytest.param("boolean_literal", "0", "FALSE"),
        pytest.param("boolean_literal", "TRUE", "TRUE"),
        pytest.param("boolean_literal", "FALSE", "FALSE"),
    ],
)
def test_bool_literal_roundtrip(name, value, expected):
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    assert str(transformed) == expected


@pytest.mark.parametrize(
    "name, value",
    [
        pytest.param("direct_variable", "%IX1.2"),
        pytest.param("direct_variable", "%IX1"),
        pytest.param("direct_variable", "%QX1.2"),
        pytest.param("direct_variable", "%QX1"),
        pytest.param("direct_variable", "%MX1.2"),
        pytest.param("direct_variable", "%MX1"),
        pytest.param("direct_variable", "%IB1.2"),
        pytest.param("direct_variable", "%IW1"),
        pytest.param("direct_variable", "%QD1.2"),
        pytest.param("direct_variable", "%QL1"),
        pytest.param("direct_variable", "%MW1.2"),
        pytest.param("direct_variable", "%ML1"),
        pytest.param("multi_element_variable", "a.b"),
        pytest.param("multi_element_variable", "a^.b"),
        pytest.param("multi_element_variable", "a^.b[1, 2]"),
        pytest.param("multi_element_variable", "a[1, 2, 3, 4]^.b[1, 2]"),
        pytest.param("multi_element_variable", "Abc123[1]^._defGhi"),
        pytest.param("expression", "1 + 1"),
        pytest.param("expression", "1 / 2"),
        pytest.param("expression", "3 * 4"),
        pytest.param("expression", "3 MOD 4"),
        pytest.param("expression", "3 XOR 4"),
        pytest.param("expression", "3 = 4"),
        pytest.param("expression", "3 <> 4"),
        pytest.param("expression", "3 <= 4"),
        pytest.param("expression", "3 >= 4"),
        pytest.param("expression", "3 ** 4"),
        pytest.param("expression", "1 + 2 * (3 - 4)"),
        pytest.param("expression", "NOT 1"),
        pytest.param("expression", "NOT (3 - 4)"),
        pytest.param("expression", "(i_xTrigger OR NOT i_xPress_OK) AND NOT xVeto"),
        pytest.param("simple_type_declaration", "TypeName : INT"),
        pytest.param("simple_type_declaration", "TypeName : INT := 5"),
        pytest.param("simple_type_declaration", "TypeName : INT := 5 + 1 * (2)"),
        pytest.param("simple_type_declaration", "TypeName : REFERENCE TO INT"),
        pytest.param("simple_type_declaration", "TypeName : POINTER TO INT"),
        pytest.param("simple_type_declaration", "TypeName EXTENDS a.b : POINTER TO INT"),
        pytest.param("subrange_type_declaration", "TypeName : INT (1..2)"),
        pytest.param("subrange_type_declaration", "TypeName : INT (*) := 1"),
        pytest.param("enumerated_type_declaration", "TypeName : TypeName := Value"),
        pytest.param("enumerated_type_declaration", "TypeName : (Value1 := 1, Value2 := 2)"),
        pytest.param("enumerated_type_declaration", "TypeName : (Value1 := 1, Value2 := 2) INT := Value1"),  # noqa: E501
    ],
)
def test_expression_roundtrip(name, value):
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into:\n\n{parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    print("\n\nTransformed:")
    print(repr(transformed))
    print("\n\nOr:")
    print(transformed)
    assert str(transformed) == value
