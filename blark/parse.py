"""
`blark parse` is a command-line utility to parse TwinCAT3 source code projects
and files.
"""
from __future__ import annotations

import argparse
import enum
import json
import pathlib
import sys
import traceback
from typing import Callable, Generator, Optional, Tuple, Union

import lark

import blark

from . import solution, summary
from . import transform as tf
from . import util
from .input import BlarkCompositeSourceItem, BlarkSourceItem, load_file_by_name
from .transform import GrammarTransformer
from .util import AnyPath

try:
    import apischema
except ImportError:
    apischema = None


DESCRIPTION = __doc__
AnyFile = Union[str, pathlib.Path]

_PARSER = None


class BlarkStartingRule(enum.Enum):
    iec_source = enum.auto()
    data_type_declaration = enum.auto()
    function_declaration = enum.auto()
    function_block_type_declaration = enum.auto()
    function_block_method_declaration = enum.auto()
    function_block_property_declaration = enum.auto()
    program_declaration = enum.auto()
    global_var_declarations = enum.auto()
    statement_list = enum.auto()
    action = enum.auto()


def new_parser(start: Optional[list[str]] = None, **kwargs) -> lark.Lark:
    """
    Get a new parser for TwinCAT flavor IEC61131-3 code.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    if start is None:
        start = [rule.name for rule in BlarkStartingRule]

    return lark.Lark.open_from_package(
        "blark",
        blark.GRAMMAR_FILENAME.name,
        parser="earley",
        maybe_placeholders=True,
        propagate_positions=True,
        start=start,
        **kwargs
    )


def get_parser() -> lark.Lark:
    """Get a cached lark.Lark parser for TwinCAT flavor IEC61131-3 code."""
    global _PARSER

    if _PARSER is None:
        _PARSER = new_parser()
    return _PARSER


DEFAULT_PREPROCESSORS = []
_DEFAULT_PREPROCESSORS = object()


ParseResult = Union[Exception, tf.SourceCode, lark.Tree]
Preprocessor = Callable[[str], str]


def parse_source_code(
    source_code: str,
    *,
    verbose: int = 0,
    fn: AnyPath = "unknown",
    preprocessors: list[Preprocessor] = _DEFAULT_PREPROCESSORS,
    parser: Optional[lark.Lark] = None,
    transform: bool = True,
    starting_rule: Optional[str] = None,
    line_map: Optional[dict[int, int]] = None,
) -> Union[tf.SourceCode, lark.Tree]:
    """
    Parse source code and return the transformed result.

    Parameters
    ----------
    source_code : str
        The source code text.

    verbose : int, optional
        Verbosity level for output.

    fn : pathlib.Path or str, optional
        The filename associated with the source code.

    preprocessors : list, optional
        Callable preprocessors to apply to the source code.

    parser : lark.Lark, optional
        The parser instance to use.  Defaults to the global shared one from
        ``get_parser``.

    transform : bool, optional
        If True, transform the output into blark-defined Python dataclasses.
        Otherwise, return the ``lark.Tree`` instance.
    """
    if preprocessors is _DEFAULT_PREPROCESSORS:
        preprocessors = DEFAULT_PREPROCESSORS

    processed_source = source_code
    for preprocessor in preprocessors:
        processed_source = preprocessor(processed_source)

    comments, processed_source = util.find_and_clean_comments(processed_source)
    if parser is None:
        parser = get_parser()

    if starting_rule is None:
        # NOTE: back-compat -> default to 'iec_source' here
        if "iec_source" in parser.options.start:
            starting_rule = "iec_source"
        else:
            starting_rule = parser.options.start[0]

    try:
        tree = parser.parse(processed_source, start=starting_rule)
    except Exception as ex:
        if verbose > 1:
            print("[Failure] Parse failure")
            print("-------------------------------")
            print(source_code)
            print("-------------------------------")
            print(f"{type(ex).__name__} {ex}")
            print(f"[Failure] {fn}")
        raise

    if line_map is not None:
        tree = util.rebuild_lark_tree_with_line_map(tree, line_map)

    if verbose > 2:
        print(f"Successfully parsed {fn}:")
        print("-------------------------------")

        if line_map is not None:
            code_lines = dict(enumerate(source_code.splitlines(), 1))
            for code_lineno, source_lineno in line_map.items():
                line = code_lines[code_lineno]
                print(f"{code_lineno} ({source_lineno}) {line}")
        else:
            for lineno, line in enumerate(source_code.splitlines(), 1):
                print(f"{lineno}: {line}")

        print("-------------------------------")
        print(tree.pretty())
        print("-------------------------------")
        print(f"[Success] End of {fn}")

    if transform:
        transformer = GrammarTransformer(
            comments=comments,
            fn=fn,
            source_code=source_code,
        )
        return transformer.transform(tree)
    return tree


def parse_single_file(
    fn: AnyPath, *,
    transform: bool = True,
    verbose: int = 0,
) -> Union[tf.SourceCode, lark.Tree]:
    """Parse a single source code file."""
    source_code = util.get_source_code(fn)
    return parse_source_code(source_code, fn=fn, transform=transform, verbose=verbose)


def parse_project(
    tsproj_project: AnyFile,
    *,
    verbose: int = 0,
    transform: bool = True
) -> Generator[Tuple[pathlib.Path, ParseResult], None, None]:
    """Parse an entire tsproj project file."""
    sol = solution.make_solution_from_files(tsproj_project)
    for item in solution.get_blark_input_from_solution(sol):
        yield from parse_item(item, verbose=verbose, transform=transform)


def parse_item(
    item: Union[BlarkSourceItem, BlarkCompositeSourceItem],
    *,
    verbose: int = 0,
    transform: bool = True,
) -> Generator[Tuple[pathlib.Path, ParseResult], None, None]:
    if isinstance(item, BlarkCompositeSourceItem):
        for part in item.parts:
            yield from parse_item(part, verbose=verbose)
        return

    code, line_map = item.get_code_and_line_map()

    # TODO: change the API to yield the sourceitem with the parse result
    try:
        filename = list(item.get_filenames())[0]
    except IndexError:
        if not item.lines:
            return
        filename = None

    yield filename, parse_source_code(
        code,
        starting_rule=item.grammar_rule,
        line_map=line_map,
        transform=transform,
    )


def parse(
    path: AnyPath,
    *,
    verbose: int = 0,
    transform: bool = True,
) -> Generator[Tuple[pathlib.Path, ParseResult], None, None]:
    """
    Parse the given source code file (or all files from the given project).
    """
    for item in load_file_by_name(path):
        # try:
        yield from parse_item(item, verbose=verbose, transform=transform)
        # except Exception as ex:
        #     tb = traceback.format_exc()
        #     ex.traceback = tb
        #     yield ex


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

    argparser.add_argument(
        "--json",
        dest="use_json",
        action="store_true",
        help="Output JSON representation only"
    )

    return argparser


def summarize(code: tf.SourceCode) -> summary.CodeSummary:
    """Get a code summary instance from a SourceCode item."""
    return summary.CodeSummary.from_source(code)


def main(
    filename: Union[str, pathlib.Path],
    verbose: int = 0,
    debug: bool = False,
    interactive: bool = False,
    summary: bool = False,
    use_json: bool = False,
):
    """
    Parse the given source code/project.
    """
    result_by_filename = {}
    failures = []
    print_filenames = sys.stdout if verbose > 0 else None
    filename = pathlib.Path(filename)

    if use_json:
        print_filenames = False

        if apischema is None:
            raise RuntimeError(
                "Optional dependency apischema is required to output a JSON "
                "representation of source code."
            )

    for fn, result in parse(filename, verbose=verbose):
        if print_filenames:
            print(f"* Loading {fn}")
        result_by_filename[fn] = result
        if isinstance(result, Exception):
            failures.append((fn, result))
            if interactive:
                util.python_debug_session(
                    namespace={"fn": fn, "result": result},
                    message=(
                        f"Failed to parse {fn}. {type(result).__name__}: {result}\n"
                        f"{result.traceback}"
                    ),
                )
            elif verbose > 1:
                print(result.traceback)
        elif summary:
            print(summarize(result_by_filename[fn]))

        if use_json:
            serialized = apischema.serialize(
                result_by_filename[fn],
                exclude_defaults=True,
                no_copy=True,
            )
            print(json.dumps(serialized, indent=2))

    if not result_by_filename:
        return {}

    if interactive:
        if len(result_by_filename) > 1:
            util.python_debug_session(
                namespace={"fn": filename, "results": result_by_filename},
                message=(
                    f"Parsed all files successfully: {list(result_by_filename)}\n"
                    f"Access all results by filename in the variable ``results``"
                )
            )
        else:
            ((filename, result),) = list(result_by_filename.items())
            util.python_debug_session(
                namespace={"fn": filename, "result": result},
                message=(
                    f"Parsed single file successfully: {filename}.\n"
                    f"Access its transformed value in the variable ``result``."
                )
            )

    if failures:
        print("Failed to parse some source code files:")
        for fn, exception in failures:
            header = f"{fn}"
            print(header)
            print("-" * len(header))
            print(f"({type(exception).__name__}) {exception}")
            print()
            # if verbose > 1:
            traceback.print_exc()

        if not debug:
            sys.exit(1)

    return result_by_filename
