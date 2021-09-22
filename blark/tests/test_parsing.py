import pathlib

import lark
import pytest

from ..parse import parse_single_file

TEST_PATH = pathlib.Path(__file__).parent


pous = list(str(path) for path in TEST_PATH.glob("**/*.TcPOU"))
additional_pous = TEST_PATH / "additional_pous.txt"

if additional_pous.exists():
    pous += open(additional_pous, "rt").read().splitlines()


@pytest.fixture(params=pous)
def pou_filename(request):
    return request.param


def test_parsing(pou_filename):
    try:
        parse_single_file(pou_filename, verbose=2)
    except FileNotFoundError:
        pytest.skip(f"Missing file: {pou_filename}")


def pytest_html_results_table_row(report, cells):
    # pytest results using pytest-html; show only failures for now:
    if report.passed:
        del cells[:]


def test_instruction_list_grammar_load():
    lark.Lark.open_from_package(
        "blark",
        "instruction_list.lark",
        parser="earley",
    )


def get_partial_grammar(name):
    return lark.Lark(
        f"""
        %import common.WS
        %ignore WS
        %import iec.{name} -> {name}
        """,
        import_paths=[TEST_PATH.parent],
        start=name,
    )


must_fail = pytest.mark.xfail(reason="Bad input", strict=True)


@pytest.mark.parametrize(
    "name, value",
    [
        pytest.param("integer_literal", "12"),
        pytest.param("integer_literal", "abc", marks=must_fail),
        pytest.param("integer_literal", "INT#12"),
        pytest.param("integer_literal", "UDINT#12"),
        pytest.param("integer_literal", "UDINT#2#010"),
        pytest.param("integer_literal", "2#10010"),
        pytest.param("integer_literal", "8#22"),
        pytest.param("integer_literal", "16#12"),
        pytest.param("single_byte_character_string", "'abc'"),
        pytest.param("double_byte_character_string", '"abc"'),
        pytest.param("single_byte_string_spec", "STRING[1]"),
        pytest.param("single_byte_string_spec", "STRING(1)"),
        pytest.param("single_byte_string_spec", "STRING(1) := 'abc'"),
        pytest.param("double_byte_string_spec", "WSTRING[1]"),
        pytest.param("double_byte_string_spec", "WSTRING(1)"),
        pytest.param("double_byte_string_spec", 'WSTRING(1) := "abc"'),
    ],
)
def test_rule_smoke(name, value):
    result = get_partial_grammar(name).parse(value)
    print(f"rule {name} value {value!r} into {result}")
