import pathlib

from ..solution import Project, Solution


def test_solution_regex():
    solution_source = r"""
        Project("{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEE0}") = "a", "a\a.tsproj", "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEE1}"
    """.strip()  # noqa: E501

    root = pathlib.Path.cwd().resolve()
    solution = Solution.from_contents(solution_source, root)
    assert solution.projects == [
        Project(
            name="a",
            solution_guid="{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEE0}",
            guid="{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEE1}",
            saved_path=pathlib.PureWindowsPath("a") / "a.tsproj",
            local_path=None,
        ),
    ]
