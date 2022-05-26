import textwrap

import pytest

from ..util import find_and_clean_comments


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
            )
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
    ]
)
def test_replace_comments(code, expected):
    expected = expected if expected is not None else code
    comments, replaced = find_and_clean_comments(code, replace_char="x")
    print(f"""
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
""")

    assert replaced == expected

    if any(code.strip().startswith(c) for c in ['"', "'"]):
        expected_comments = []
    else:
        expected_comments = [textwrap.dedent(code.lstrip("\n")).strip()]
    assert [str(comment) for comment in comments] == expected_comments
    print("comments=", comments)
