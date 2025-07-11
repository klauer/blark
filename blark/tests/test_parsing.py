import pathlib

import lark
import pytest

from ..parse import parse, parse_source_code, summarize
from ..util import SourceType
from . import conftest

TEST_PATH = pathlib.Path(__file__).parent


def test_parsing_tcpous(twincat_pou_filename: str):
    """Test parsing TwinCAT TcPOU files."""
    all_parts = []
    for part in parse(twincat_pou_filename):
        all_parts.append(part)

        transformed = part.transform()
        print("transformed:")
        print(transformed)
        print("summary:")
        if part.exception:
            raise part.exception
        conftest.check_serialization(transformed, deserialize=False)

    print(summarize(all_parts))


def test_parsing_source(source_filename: str):
    """Test plain source 61131 files."""
    with open(source_filename, "r", encoding="utf-8") as src:
        content = src.read()

    result = parse_source_code(content)
    transformed = result.transform()
    print("transformed:")
    print(transformed)
    print("summary:")
    if result.exception:
        raise result.exception

    if result.item.type in (
        SourceType.dut,
        SourceType.function,
        SourceType.function_block,
        SourceType.interface,
        SourceType.statement_list,
        SourceType.var_global,
        SourceType.program,
    ):
        print(summarize([result]))
    conftest.check_serialization(transformed, deserialize=False)


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


@pytest.mark.parametrize(
    ("code", "expected_snippets"),
    [
        pytest.param(
            """\
(*
123456789012345678
*)
// comment
{pragma}
VAR_GLOBAL
    dummy : BOOL;
END_VAR
""",
            [
                "(*\n123456789012345678\n*)",
                "// comment",
                "{pragma}",
            ],
            id="issue_109",
        ),
    ],
)
def test_comment_parsing(code: str, expected_snippets: list[str]):
    result = parse_source_code(code)

    print(result.comments)

    snippets = [
        result.source_code[token.start_pos: token.end_pos] for token in result.comments
    ]
    assert snippets == expected_snippets


def tree_contains_token(tree, token):
    """
    Checks whether the given lark tree contains the given token, either as a node
    or as a leaf.
    """
    return any((
        tree.data == token,
        token in tree.children,
        any(tree_contains_token(child, token)
            for child in tree.children
            if isinstance(child, lark.Tree))))


@pytest.mark.parametrize(
    "start_rule, code, token_type, token_value",
    [
        pytest.param("unary_expression", "TRUE", "RULE", "constant"),
        pytest.param("unary_expression", "TRUE", "TRUE_VALUE", "TRUE"),
        pytest.param("unary_expression", "FALSE", "RULE", "constant"),
        pytest.param("unary_expression", "FALSE", "FALSE_VALUE", "FALSE"),
    ],
)
def test_key_token(start_rule: str, code: str, token_type: str, token_value: str):
    """Test whether the parse tree contains the key token."""
    result = conftest.get_grammar(start=start_rule).parse(code)
    assert tree_contains_token(result, lark.Token(token_type, token_value))
