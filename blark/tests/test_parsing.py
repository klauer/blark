import pathlib

import pytest

from ..parse import parse, parse_source_code, summarize
from .conftest import get_grammar

TEST_PATH = pathlib.Path(__file__).parent


pous = list(str(path) for path in TEST_PATH.glob("**/*.TcPOU"))
additional_pous = TEST_PATH / "additional_pous.txt"

if additional_pous.exists():
    pous += open(additional_pous, "rt").read().splitlines()

sources = list(str(path) for path in TEST_PATH.glob("**/*.st"))


@pytest.fixture(params=pous)
def pou_filename(request):
    return request.param


@pytest.fixture(params=sources)
def source_filename(request):
    return request.param


def test_parsing_tcpous(pou_filename):
    try:
        ((_, result),) = list(parse(pou_filename))
    except FileNotFoundError:
        pytest.skip(f"Missing file: {pou_filename}")
    else:
        print("transformed:")
        print(result)
        print("summary:")
        if isinstance(result, Exception):
            raise result
        print(summarize(result))


def test_parsing_source(source_filename):
    """Test plain source 61131 files."""
    try:
        with open(source_filename, "r", encoding="utf-8") as src:
            content = src.read()
        result = parse_source_code(content)
    except FileNotFoundError:
        pytest.skip(f"Missing file: {source_filename}")
    else:
        print("transformed:")
        print(result)
        print("summary:")
        if isinstance(result, Exception):
            raise result
        print(summarize(result))


def pytest_html_results_table_row(report, cells):
    # pytest results using pytest-html; show only failures for now:
    if report.passed:
        del cells[:]


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
    result = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {result}")
