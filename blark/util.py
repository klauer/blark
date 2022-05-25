import hashlib
import pathlib
import re
from typing import Any, Dict, Generator, List, Tuple, Union

import lark
import pytmc
import pytmc.parser

RE_LEADING_WHITESPACE = re.compile('^[ \t]+', re.MULTILINE)
AnyPath = Union[str, pathlib.Path]

TWINCAT_PROJECT_FILE_EXTENSIONS = {".tsproj"}
TWINCAT_SOURCE_EXTENSIONS = {
    ".tcdut",  # data unit type
    # ".tcgtlo",  # global text list object
    ".tcgvl",  # global variable list
    # ".tcipo",  # image pool
    ".tcpou",  # program organization unit
    # ".tcrmo",  # recipe manager
    # ".tctlo",  # text list object
    # ".tctto",  # task object
    # ".tcvis",  # visualization
    # ".tcvmo",  # visualization manager object
    # ".tmc",  # module class - description of project
    # ".tpy",  # tmc-like inter-vendor format
    # ".xti",  # independent project file
}


def get_source_code(fn: AnyPath, *, encoding: str = "utf-8") -> str:
    """
    Get source code from the given file.

    Supports TwinCAT source files (in XML format) or plain text files.

    Parameters
    ----------
    fn : str or pathlib.Path
        The path to the source code file.

    encoding : str, optional, keyword-only
        The encoding to use when opening the file.  Defaults to utf-8.

    Returns
    -------
    str
        The source code.

    Raises
    ------
    FileNotFoundError
        If ``fn`` does not point to a valid file.

    ValueError
        If a TwinCAT file is specified but no source code is associated with
        it.
    """
    fn = pathlib.Path(fn)

    if fn.suffix.lower() not in TWINCAT_SOURCE_EXTENSIONS:
        with open(fn, "rt", encoding=encoding) as fp:
            return fp.read()

    root = pytmc.parser.parse(fn)

    for item in root.find(pytmc.parser.TwincatItem):
        get_source = getattr(item, "get_source_code", None)
        if get_source is not None:
            return get_source()

    raise ValueError(
        "Unable to find pytmc TwincatItem with source code "
        "(i.e., with `get_source_code` as an attribute)"
    )


def indent_inner(text: str, prefix: str) -> str:
    """Indent the inner lines of ``text`` (not first and last) with ``prefix``."""
    lines = text.splitlines()
    if len(lines) < 3:
        return text

    return "\n".join(
        (
            lines[0],
            *(f"{prefix}{line}" for line in lines[1:-1]),
            lines[-1],
        )
    )


def python_debug_session(namespace: Dict[str, Any], message: str):
    """
    Enter an interactive debug session with pdb or IPython, if available.
    """
    import blark  # noqa

    debug_namespace = dict(pytmc=pytmc, blark=blark)
    debug_namespace.update(
        **{k: v for k, v in namespace.items()
           if not k.startswith('__')}
    )
    globals().update(debug_namespace)

    print(
        "\n".join(
            (
                "-- blark debug --",
                message,
                "-- blark debug --",
            )
        )
    )

    try:
        from IPython import embed  # noqa
    except ImportError:
        import pdb  # noqa
        pdb.set_trace()
    else:
        embed()


def find_and_clean_comments(
    text: str, *, replace_char: str = " "
) -> Tuple[List[lark.Token], str]:
    """
    Clean nested multiline comments from ``text``.

    For a nested comment like ``"(* (* abc *) *)"``, the inner comment markers
    would be replaced with ``replace_char``, resulting in the return value
    ``"(*    abc    *)"``.
    """
    lines = text.splitlines()
    original_lines = list(lines)
    multiline_comments = []
    in_single_comment = False
    in_single_quote = False
    in_double_quote = False
    pragma_state = []
    skip = 0
    NEWLINES = "\n\r"
    SINGLE_COMMENT = "//"
    OPEN_COMMENT = "(*"
    CLOSE_COMMENT = "*)"
    OPEN_PRAGMA = "{"
    CLOSE_PRAGMA = "}"

    comments_and_pragmas: List[lark.Token] = []

    def get_characters() -> Generator[Tuple[int, int, str, str], None, None]:
        """Yield line information and characters."""
        for lineno, line in enumerate(text.splitlines()):
            colno = 0
            for colno, (this_ch, next_ch) in enumerate(zip(line, line[1:] + "\n")):
                yield lineno, colno, this_ch, next_ch
            yield lineno, colno, "\n", ""

    def fix_line(lineno: int, colno: int) -> str:
        """Uncomment a nested multiline comment at (line, col)."""
        replacement_line = list(lines[lineno])
        replacement_line[colno] = replace_char
        replacement_line[colno + 1] = replace_char
        return "".join(replacement_line)

    def get_token(start_line: int, start_col: int, end_line: int, end_col: int) -> lark.Token:
        if start_line != end_line:
            block = "\n".join(
                (
                    original_lines[start_line][start_col:],
                    *original_lines[start_line + 1:end_line],
                    original_lines[end_line][:end_col + 1]
                )
            )
        else:
            block = original_lines[start_line][start_col:end_col + 1]

        if block.startswith("//"):
            type_ = "SINGLE_LINE_COMMENT"
        elif block.startswith("(*"):  # *)
            type_ = "MULTI_LINE_COMMENT"
        elif block.startswith("{"):  # }
            type_ = "PRAGMA"
        else:
            raise RuntimeError("Unexpected block: {contents}")

        if start_line != end_line:
            # TODO: move "*)" to separate line
            block = indent_inner(
                RE_LEADING_WHITESPACE.sub("", block),
                prefix={
                    "SINGLE_LINE_COMMENT": "",   # this would be a bug
                    "MULTI_LINE_COMMENT": "    ",
                    "PRAGMA": "    ",
                }[type_],
            )

        return lark.Token(
            type_,
            block,
            line=start_line + 1, end_line=end_line + 1,
            column=start_col + 1, end_column=end_col + 1,
        )

    for lineno, colno, this_ch, next_ch in get_characters():
        if skip:
            skip -= 1
            continue

        if in_single_comment:
            in_single_comment = this_ch not in NEWLINES
            continue

        pair = this_ch + next_ch
        if not in_single_quote and not in_double_quote:
            if this_ch == OPEN_PRAGMA and not multiline_comments:
                pragma_state.append((lineno, colno))
                continue
            if this_ch == CLOSE_PRAGMA and not multiline_comments:
                start_line, start_col = pragma_state.pop(-1)
                if len(pragma_state) == 0:
                    comments_and_pragmas.append(
                        get_token(start_line, start_col, lineno, colno + 1)
                    )
                continue

            if pragma_state:
                continue

            if pair == OPEN_COMMENT:
                multiline_comments.append((lineno, colno))
                skip = 1
                if len(multiline_comments) > 1:
                    # Nested multi-line comment
                    lines[lineno] = fix_line(lineno, colno)
                continue
            if pair == CLOSE_COMMENT:
                start_line, start_col = multiline_comments.pop(-1)
                if len(multiline_comments) > 0:
                    # Nested multi-line comment
                    lines[lineno] = fix_line(lineno, colno)
                else:
                    comments_and_pragmas.append(
                        get_token(start_line, start_col, lineno, colno + 1)
                    )
                skip = 1
                continue
            if pair == SINGLE_COMMENT:
                in_single_comment = True
                comments_and_pragmas.append(
                    get_token(
                        lineno, colno, lineno, len(lines[lineno])
                    )
                )
                continue

        if not multiline_comments and not in_single_comment:
            if pair == "$'" and in_single_quote:
                # This is an escape for single quotes
                skip = 1
            elif pair == '$"' and in_double_quote:
                # This is an escape for double quotes
                skip = 1
            elif this_ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif this_ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif pair == SINGLE_COMMENT:
                in_single_comment = True

    if multiline_comments or in_single_quote or in_double_quote:
        # Syntax error in source? Return the original and let lark fail
        return comments_and_pragmas, text

    return comments_and_pragmas, "\n".join(lines)


def remove_comment_characters(text: str) -> str:
    """Take only the inner contents of a given comment."""
    text = text.strip()
    if text.startswith("/"):
        return text.lstrip("/ ")
    return text.strip("()").strip("* ")


def get_tsprojects_from_filename(
    filename: AnyPath,
) -> Tuple[pathlib.Path, List[pathlib.Path]]:
    """
    From a TwinCAT solution (.sln) or .tsproj, return all tsproj projects.

    Returns
    -------
    root : pathlib.Path
        Project root directory (where the solution or provided tsproj is
        located).

    projects : list of pathlib.Path
        List of tsproj projects paths.
    """
    abs_path = pathlib.Path(filename).resolve()
    if abs_path.suffix == '.tsproj':
        return abs_path.parent, [abs_path]
    if abs_path.suffix == '.sln':
        return abs_path.parent, pytmc.parser.projects_from_solution(abs_path)

    raise RuntimeError(f'Expected a .tsproj/.sln file; got {abs_path.suffix!r}')


def get_file_sha256(filename: AnyPath) -> str:
    """Hash a file's contents with the SHA-256 algorithm."""
    with open(filename, "rb") as fp:
        return hashlib.sha256(fp.read()).hexdigest()
