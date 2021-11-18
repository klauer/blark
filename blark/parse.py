"""
`blark parse` is a command-line utility to parse TwinCAT3 source code
files in conjunction with pytmc.
"""
import argparse
import pathlib
import re
import sys
from typing import Generator, List, Optional, Tuple, Union

import lark
import pytmc

import blark

from . import transform as tf
from .transform import GrammarTransformer
from .util import get_source_code, indent_inner, python_debug_session

DESCRIPTION = __doc__
RE_LEADING_WHITESPACE = re.compile('^[ \t]+', re.MULTILINE)

_PARSER = None


def new_parser(**kwargs) -> lark.Lark:
    """
    Get a new parser for TwinCAT flavor IEC61131-3 code.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return lark.Lark.open_from_package(
        "blark",
        blark.GRAMMAR_FILENAME.name,
        parser="earley",
        maybe_placeholders=True,
        propagate_positions=True,
        **kwargs
    )


def get_parser() -> lark.Lark:
    """Get a cached lark.Lark parser for TwinCAT flavor IEC61131-3 code."""
    global _PARSER

    if _PARSER is None:
        _PARSER = new_parser()
    return _PARSER


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


DEFAULT_PREPROCESSORS = []
_DEFAULT_PREPROCESSORS = object()


def parse_source_code(
    source_code: str,
    *,
    verbose: int = 0,
    fn: str = "unknown",
    preprocessors=_DEFAULT_PREPROCESSORS,
    parser: Optional[lark.Lark] = None,
):
    """
    Parse source code and return the transformed result.

    Parameters
    ----------
    source_code : str
        The source code text.

    verbose : int, optional
        Verbosity level for output.

    fn : str, optional
        The filename associated with the source code.

    preprocessors : list, optional
        Callable preprocessors to apply to the source code.

    parser : lark.Lark, optional
        The parser instance to use.  Defaults to the global shared one from
        ``get_parser``.
    """
    if preprocessors is _DEFAULT_PREPROCESSORS:
        preprocessors = DEFAULT_PREPROCESSORS

    processed_source = source_code
    for preprocessor in preprocessors:
        processed_source = preprocessor(processed_source)

    comments, processed_source = find_and_clean_comments(processed_source)
    if parser is None:
        parser = get_parser()

    try:
        tree = parser.parse(processed_source)
    except Exception as ex:
        if verbose > 1:
            print("[Failure] Parse failure")
            print("-------------------------------")
            print(source_code)
            print("-------------------------------")
            print(f"{type(ex).__name__} {ex}")
            print(f"[Failure] {fn}")
        raise

    if verbose > 2:
        print(f"Successfully parsed {fn}:")
        print("-------------------------------")
        print(source_code)
        print("-------------------------------")
        print(tree.pretty())
        print("-------------------------------")
        print(f"[Success] End of {fn}")

    return GrammarTransformer(comments).transform(tree)


def parse_single_file(fn, *, verbose=0):
    "Parse a single source code file"
    source_code = get_source_code(fn)
    return parse_source_code(source_code, fn=fn, verbose=verbose)


def parse_project(tsproj_project, *, print_filenames=None, verbose=0):
    "Parse an entire tsproj project file"
    proj_path = pathlib.Path(tsproj_project)
    proj_root = proj_path.parent.resolve().absolute()  # noqa: F841 TODO

    if proj_path.suffix.lower() not in (".tsproj",):
        raise ValueError("Expected a .tsproj file")

    project = pytmc.parser.parse(proj_path)
    results = {}
    success = True
    for i, plc in enumerate(project.plcs, 1):
        source_items = (
            list(plc.dut_by_name.items())
            + list(plc.gvl_by_name.items())
            + list(plc.pou_by_name.items())
        )
        for name, source_item in source_items:
            if not hasattr(source_item, "get_source_code"):
                continue

            if print_filenames is not None:
                print(
                    f"* Parsing {source_item.filename}",
                    file=print_filenames
                )
            source_code = source_item.get_source_code()
            if not source_code:
                continue

            try:
                results[name] = parse_source_code(
                    source_code, fn=source_item.filename, verbose=verbose
                )
            except Exception as ex:
                results[name] = ex
                ex.filename = source_item.filename
                success = False

    return success, results


def build_arg_parser(argparser=None):
    if argparser is None:
        argparser = argparse.ArgumentParser()

    argparser.description = DESCRIPTION
    argparser.formatter_class = argparse.RawTextHelpFormatter

    argparser.add_argument(
        "filename",
        type=str,
        help=(
            "Path to project, solution, source code file (.tsproj, .sln, "
            ".TcPOU, .TcGVL)"
        ),
    )

    argparser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity, up to -vvv",
    )

    argparser.add_argument(
        "--debug", action="store_true",
        help="On failure, still return the results tree"
    )

    argparser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Enter IPython (or Python) to explore source trees"
    )

    argparser.add_argument(
        "-s", "--summary", action="store_true",
        help="Summarize code inputs and outputs"
    )

    return argparser


def summarize(code):
    if isinstance(code, tf.SourceCode):
        for item in code.items:
            summarize(item)
    elif isinstance(code, tf.FunctionBlock):
        try:
            comments = code.meta.comments
        except AttributeError:
            comments = []
        print("Function block", code.name, comments)
        for decl in code.declarations:
            print(type(decl).__name__, ":")
            for item in decl.items:
                try:
                    comments = item.meta.comments
                except AttributeError:
                    comments = []

                # OK, a bit lazy for now
                try:
                    spec = item.init.spec
                except AttributeError:
                    spec = "?"

                print(
                    "\t",
                    " ".join(getattr(var, "name", var) for var in item.variables),
                    f"({spec}) {comments}"
                )


def main(
    filename: Union[str, pathlib.Path],
    verbose: int = 0,
    debug: bool = False,
    interactive: bool = False,
    summary: bool = False,
):
    """
    Parse the given source code/project.
    """
    path = pathlib.Path(filename)
    project_fns = []
    source_fns = []
    if path.suffix.lower() in (".tsproj",):
        project_fns = [path]
    elif path.suffix.lower() in (".sln",):
        project_fns = pytmc.parser.projects_from_solution(path)
    elif path.suffix.lower() in (".tcpou", ".tcgvl", ".tcdut"):
        source_fns = [path]
    else:
        raise ValueError(f"Expected a tsproj or sln file, got: {path.suffix}")

    results = {}
    success = True
    print_filenames = sys.stdout if verbose > 0 else None

    for fn in project_fns:
        if print_filenames:
            print(f"* Loading project {fn}")
        success, results[fn] = parse_project(
            fn, print_filenames=print_filenames, verbose=verbose
        )

    for fn in source_fns:
        if print_filenames:
            print(f"* Parsing {fn}")
        try:
            results[fn] = parse_single_file(fn, verbose=verbose)
        except Exception as ex:
            success = False
            results[fn] = ex
            if interactive:
                python_debug_session(
                    namespace={"fn": fn, "result": ex},
                    message=f"Failed to parse {fn}. {type(ex).__name__}: {ex}"
                )
        else:
            if summary:
                summarize(results[fn])
            if interactive:
                python_debug_session(
                    namespace={"fn": fn, "result": results[fn]},
                    message=f"Parsed {fn} successfully."
                )

    def find_failures(res):
        for name, item in res.items():
            if isinstance(item, Exception):
                yield name, item
            elif isinstance(item, dict):
                yield from find_failures(item)

    if not success:
        print("Failed to parse all source code files:")
        failures = list(find_failures(results))
        for name, item in failures:
            fn = f"[{item.filename}] " if hasattr(item, "filename") else ""
            header = f"{fn}{name}"
            print(header)
            print("-" * len(header))
            print(f"({type(item).__name__}) {item}")
            print()

        if not debug:
            sys.exit(1)

    return results
