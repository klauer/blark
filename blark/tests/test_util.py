import pytest

from ..parse import replace_comments


@pytest.mark.parametrize(
    "code, expected",
    [
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
    ]
)
def test_replace_comments(code, expected):
    expected = expected if expected is not None else code
    replaced = replace_comments(code, replace_char="x")
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
