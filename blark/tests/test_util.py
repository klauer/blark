import pathlib
import textwrap

import pytest

from .. import util
from ..input import BlarkCompositeSourceItem, BlarkSourceItem
from ..util import SourceType, find_and_clean_comments


@pytest.mark.parametrize(
    "code, expected",
    [
        pytest.param(
            """
            // abc
            """,
            None,
            id="simple_single_line",
        ),
        pytest.param(
            """
            // (* abc *)
            """,
            None,
            id="multi_in_single",
        ),
        pytest.param(
            """
            (* abc *)
            """,
            """
            (* abc *)
            """,
            id="simple",
        ),
        pytest.param(
            """
            (* (* abc *) *)
            """,
            """
            (* xx abc xx *)
            """,
            id="nested_2",
        ),
        pytest.param(
            """
            (*
             (*
                 (* abc *)
              *)
            *)
            """,
            """
            (*
             xx
                 xx abc xx
              xx
            *)
            """,
            id="nested_3",
            marks=pytest.mark.xfail(
                reason="Comments replaced OK; whitespace handling needs work"
            ),
        ),
        pytest.param(
            """
            "(* quoted *)"
            """,
            None,
            id="double_quoted",
        ),
        pytest.param(
            """
            "(* $" quoted *)"
            """,
            None,
            id="double_quoted_with_escape",
        ),
        pytest.param(
            """
            '(* quoted *)'
            """,
            None,
            id="single_quoted",
        ),
        pytest.param(
            """
            '(* $' quoted *)'
            """,
            None,
            id="single_quoted_with_escape",
        ),
        pytest.param(
            """
            {pragma}
            """,
            None,
            id="pragma",
        ),
        pytest.param(
            """
            {pragma'}
            """,
            None,
            id="pragma_mismatch_quote",
        ),
        pytest.param(
            """
            {'"pragma'}
            """,
            None,
            id="pragma_mismatch_quote",
        ),
        pytest.param(
            """
            (* {'"pragma'} *)
            """,
            None,
            id="commented_pragma",
        ),
        pytest.param(
            """
            // (* {'"pragma'} *)
            """,
            None,
            id="commented_pragma",
        ),
        pytest.param(
            """
            // {'"pragma'}
            """,
            None,
            id="commented_pragma",
        ),
        pytest.param(
            """
            {pragma (* comment *)}
            """,
            None,
            id="pragma_comment",
        ),
        pytest.param(
            """
            {pragma // comment}
            """,
            None,
            id="pragma_comment",
        ),
        pytest.param(
            """
            {pragma (* // comment *)}
            """,
            None,
            id="pragma_comment_why",
        ),
    ],
)
def test_replace_comments(code, expected):
    expected = expected if expected is not None else code
    comments, replaced = find_and_clean_comments(code, replace_char="x")
    print(
        f"""
code:
-----
{code}
-----
replaced:
-----
{replaced}
-----
expected:
-----
{expected}
-----
"""
    )

    assert replaced == expected

    if any(code.strip().startswith(c) for c in ['"', "'"]):
        expected_comments = []
    else:
        expected_comments = [textwrap.dedent(code.lstrip("\n")).strip()]
    assert [str(comment) for comment in comments] == expected_comments
    print("comments=", comments)


@pytest.mark.parametrize(
    "source, file_line, blark_line, line_map",
    [
        ("a\nb\nc", 1, 1, {1: 1, 2: 2, 3: 3}),
        ("a\nb\nc\nd", 10, 1, {1: 10, 2: 11, 3: 12, 4: 13}),
        ("a\nb\nc", 1, 5, {5: 1, 6: 2, 7: 3}),
        ("a\nb\nc\nd", 10, 5, {5: 10, 6: 11, 7: 12, 8: 13}),
    ],
)
def test_line_map(
    source: str,
    file_line: int,
    blark_line: int,
    line_map: dict[int, int],
):
    assert (
        util._build_source_to_file_line_map(file_line, blark_line, source) == line_map
    )


def test_line_map_composite():
    composite = BlarkCompositeSourceItem(
        type=SourceType.method,
        parts=[
            BlarkSourceItem(
                file=pathlib.Path("FB_DummyHA.TcPOU"),
                line=64,
                type=SourceType.method,
                code=(
                    "\n"
                    "METHOD RequestBP : BOOL\n"
                    "VAR_INPUT\n"
                    "\t(*StateID of state requesting beam parameter set*)\n"
                    "\tnReqID\t: DWORD;\n"
                    "\t(*Requested beam params*)\n"
                    "\tstReqBP\t: ST_BeamParams;\n"
                    "END_VAR\n"
                ),
                implicit_end=None,
                grammar_rule="function_block_method_declaration",
            ),
            BlarkSourceItem(
                file=pathlib.Path("FB_DummyHA.TcPOU"),
                line=74,
                type=SourceType.method,
                code="RequestBP := TRUE;",
                implicit_end=None,
                grammar_rule="statement_list",
            ),
        ],
        implicit_end="END_METHOD",
        grammar_rule="function_block_method_declaration",
    )

    code, line_map = composite.get_code(include_end=True)
    print(code)
    expected_code = (
        "\n"
        "METHOD RequestBP : BOOL\n"
        "VAR_INPUT\n"
        "\t(*StateID of state requesting beam parameter set*)\n"
        "\tnReqID\t: DWORD;\n"
        "\t(*Requested beam params*)\n"
        "\tstReqBP\t: ST_BeamParams;\n"
        "END_VAR\n"
        "RequestBP := TRUE;\n"  # <- line 9 is from line 74
        "END_METHOD"  # <- line 10 is made up
    )
    assert code == expected_code

    assert line_map == {
        1: 64,
        2: 65,
        3: 66,
        4: 67,
        5: 68,
        6: 69,
        7: 70,
        8: 71,
        9: 74,
        10: 74,
    }
