import pathlib

import pytest

from ..parse import parse, parse_source_code, summarize
from . import conftest

TEST_PATH = pathlib.Path(__file__).parent


def test_parsing_tcpous(twincat_pou_filename: str):
    """Test parsing TwinCAT TcPOU files."""
    ((_, result),) = list(parse(twincat_pou_filename))
    print("transformed:")
    print(result)
    print("summary:")
    if isinstance(result, Exception):
        raise result
    print(summarize(result))

    conftest.check_serialization(result, deserialize=False)


def test_parsing_source(source_filename: str):
    """Test plain source 61131 files."""
    with open(source_filename, "r", encoding="utf-8") as src:
        content = src.read()

    result = parse_source_code(content)
    print("transformed:")
    print(result)
    print("summary:")
    if isinstance(result, Exception):
        raise result
    print(summarize(result))
    conftest.check_serialization(result, deserialize=False)


must_fail = pytest.mark.xfail(reason="Bad input", strict=True)


@pytest.mark.parametrize(
    "name, value",
    [
        pytest.param("integer_literal", "12"),
        pytest.param("integer_literal", "abc", marks=must_fail),
        pytest.param("integer_literal", "INT#12"),
        pytest.param("integer_literal", "UDINT#12"),
        pytest.param("integer_literal", "UDINT#2#010"),
        pytest.param("integer_literal", "UDINT#2#1001_0011"),
        pytest.param("integer_literal", "DINT#16#C0FFEE"),
        pytest.param("integer_literal", "2#10010"),
        pytest.param("integer_literal", "8#22"),
        pytest.param("integer_literal", "16#12"),
        pytest.param("constant", "'abc'"),
        pytest.param("constant", '"abc"'),
        pytest.param("single_byte_string_spec", "STRING[1]"),
        pytest.param("single_byte_string_spec", "STRING(1)"),
        pytest.param("single_byte_string_spec", "STRING(1) := 'abc'"),
        pytest.param("double_byte_string_spec", "WSTRING[1]"),
        pytest.param("double_byte_string_spec", "WSTRING(1)"),
        pytest.param("double_byte_string_spec", 'WSTRING(1) := "abc"'),
        pytest.param("duration", "TIME#1D"),
        pytest.param("duration", "TIME#1S"),
        pytest.param("duration", "TIME#10S"),
        pytest.param("duration", "TIME#1H"),
        pytest.param("duration", "TIME#1M"),
        pytest.param("duration", "TIME#10MS"),
        pytest.param("duration", "TIME#1H1M1S1MS"),
        pytest.param("duration", "TIME#1.1D"),
        pytest.param("duration", "TIME#1.1S"),
        pytest.param("duration", "TIME#1.10S"),
        pytest.param("duration", "TIME#1.1H"),
        pytest.param("duration", "TIME#1.1M"),
        pytest.param("duration", "TIME#1.10MS"),
        pytest.param("duration", "TIME#1H1M1S1MS"),
        pytest.param("duration", "TIME#1.1H1M1S1MS", marks=must_fail),
    ],
)
def test_rule_smoke(grammar, name, value):
    result = conftest.get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {result}")
