from __future__ import annotations

import pathlib
import textwrap

import pytest

from ..input import BlarkCompositeSourceItem, BlarkSourceItem, BlarkSourceLine
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


def build_lines(
    filename: pathlib.Path, code: str, lineno: int
) -> list[BlarkSourceLine]:
    return [
        BlarkSourceLine(filename=filename, lineno=lineno, code=line)
        for lineno, line in enumerate(code.splitlines(), start=lineno)
    ]


def test_line_map_composite():
    filename = pathlib.Path("FB_DummyHA.TcPOU")
    method_lines = build_lines(
        lineno=64,
        filename=filename,
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
    )

    method_lines.extend(
        build_lines(
            filename=filename,
            lineno=74,
            code="RequestBP := TRUE;",
        )
    )

    composite = BlarkCompositeSourceItem(
        filename=filename,
        identifier="FB_DummyHA",
        parts=[
            BlarkSourceItem(
                identifier="FB_DummyHA.RequestBP",
                type=SourceType.method,
                lines=method_lines,
                implicit_end="END_METHOD",
                grammar_rule="function_block_method_declaration",
            ),
        ],
    )

    code, line_map = composite.get_code_and_line_map(include_end=True)
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
