"""
`blark parse` is a command-line utility to parse TwinCAT3 source code projects
and files.
"""
from __future__ import annotations

import argparse
import enum
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Generator, Optional, Sequence, Type, TypeVar, Union

import lark

import blark

from . import solution, summary
from . import transform as tf
from . import util
from .input import BlarkCompositeSourceItem, BlarkSourceItem, load_file_by_name
from .typing import Preprocessor
from .util import AnyPath

try:
    import apischema
except ImportError:
    apischema = None


logger = logging.getLogger(__name__)


DESCRIPTION = __doc__
AnyFile = Union[str, pathlib.Path]

_PARSER = None


class BlarkStartingRule(enum.Enum):
    iec_source = enum.auto()
    action = enum.auto()
    data_type_declaration = enum.auto()
    function_block_method_declaration = enum.auto()
    function_block_property_declaration = enum.auto()
    function_block_type_declaration = enum.auto()
    function_declaration = enum.auto()
    global_var_declarations = enum.auto()
    interface_declaration = enum.auto()
    program_declaration = enum.auto()
    statement_list = enum.auto()


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
        **kwargs,
    )


def get_parser() -> lark.Lark:
    """Get a cached lark.Lark parser for TwinCAT flavor IEC61131-3 code."""
    global _PARSER

    if _PARSER is None:
        _PARSER = new_parser()
    return _PARSER


DEFAULT_PREPROCESSORS = tuple()


@dataclass
class ParseResult:
    source_code: str
    item: BlarkSourceItem
    processed_source_code: str
    comments: list[lark.Token]
    line_map: Optional[dict[int, int]] = None
    filename: Optional[pathlib.Path] = None
    exception: Optional[Exception] = None
    tree: Optional[lark.Tree] = None
    parent: Optional[BlarkCompositeSourceItem] = None
    transformed: Optional[tf.SourceCode] = None

    @property
    def identifier(self) -> Optional[str]:
        if self.item is None:
            return None
        return self.item.identifier

    def transform(self) -> tf.SourceCode:
        if self.tree is None:
            raise ValueError(
                f"Source code was not successfully parsed; cannot transform. "
                f"Exception was {type(self.exception).__name__}: {self.exception}"
            )

        if self.transformed is None:
            self.transformed = tf.transform(
                source_code=self.source_code,
                tree=self.tree,
                comments=self.comments,
                line_map=self.line_map,
                filename=self.filename,
            )
        return self.transformed

    def dump_source(self, fp=sys.stdout) -> None:
        if self.line_map is not None:
            # NOTE: split("\n") and splitlines() differ wrt the final newline
            code_lines = dict(enumerate(self.source_code.split("\n"), 1))
            for code_lineno, source_lineno in self.line_map.items():
                line = code_lines[code_lineno]
                print(f"{code_lineno} ({source_lineno}) {line}", file=fp)
        else:
            for lineno, line in enumerate(self.source_code.splitlines(), 1):
                print(f"{lineno}: {line}", file=fp)


def parse_source_code(
    source_code: str,
    *,
    verbose: int = 0,
    fn: AnyPath = "unknown",
    preprocessors: Sequence[Preprocessor] = DEFAULT_PREPROCESSORS,
    parser: Optional[lark.Lark] = None,
    starting_rule: Optional[str] = None,
    line_map: Optional[dict[int, int]] = None,
    item: Optional[BlarkSourceItem] = None,
) -> ParseResult:
    """
    Parse source code into a ``ParseResult``.

    Parameters
    ----------
    source_code : str
        The source code text.

    verbose : int, optional
        Verbosity level for output. (deprecated)

    fn : pathlib.Path or str, optional
        The filename associated with the source code.

    preprocessors : list, optional
        Callable preprocessors to apply to the source code.

    parser : lark.Lark, optional
        The parser instance to use.  Defaults to the global shared one from
        ``get_parser``.
    """
    processed_source = source_code
    for preprocessor in preprocessors:
        processed_source = preprocessor(processed_source)

    comments, processed_source = util.find_and_clean_comments(
        processed_source,
        line_map=line_map,
    )
    if parser is None:
        parser = get_parser()

    if starting_rule is None:
        # NOTE: back-compat -> default to 'iec_source' here
        if "iec_source" in parser.options.start:
            starting_rule = "iec_source"
        else:
            starting_rule = parser.options.start[0]

    if item is None:
        item = BlarkSourceItem.from_code(
            source_code,
            grammar_rule=starting_rule,
        )

    result = ParseResult(
        item=item,
        source_code=source_code,
        processed_source_code=processed_source,
        line_map=line_map,
        comments=comments,
        filename=pathlib.Path(fn),
    )

    try:
        result.tree = parser.parse(processed_source, start=starting_rule)
    except lark.UnexpectedInput as ex:
        if line_map:
            ex.line = line_map.get(ex.line, ex.line)
        result.exception = ex
    except lark.LarkError as ex:
        result.exception = ex

    return result


def parse_single_file(fn: AnyPath, **kwargs) -> ParseResult:
    """Parse a single source code file."""
    source_code = util.get_source_code(fn)
    return parse_source_code(source_code, fn=fn, **kwargs)


def parse_project(
    tsproj_project: AnyFile,
    **kwargs,
) -> Generator[ParseResult, None, None]:
    """Parse an entire tsproj project file."""
    sol = solution.make_solution_from_files(tsproj_project)
    for item in solution.get_blark_input_from_solution(sol):
        yield from parse_item(item, **kwargs)


def parse_item(
    item: Union[BlarkSourceItem, BlarkCompositeSourceItem],
    **kwargs,
) -> Generator[ParseResult, None, None]:
    if isinstance(item, BlarkCompositeSourceItem):
        for part in item.parts:
            for res in parse_item(part, **kwargs):
                res.filename = res.filename or item.filename
                res.parent = item
                yield res
        return

    code, line_map = item.get_code_and_line_map()

    try:
        filename = list(item.get_filenames())[0]
    except IndexError:
        if not item.lines:
            return
        filename = None

    result = parse_source_code(
        code,
        starting_rule=item.grammar_rule,
        line_map=line_map,
        fn=filename or "unknown",
        **kwargs,
    )
    result.item = item
    yield result


def parse(
    path: AnyPath,
    input_format: Optional[str] = None,
    **kwargs,
) -> Generator[ParseResult, None, None]:
    """
    Parse the given source code file (or all files from the given project).
    """
    for item in load_file_by_name(path, input_format=input_format):
        yield from parse_item(item, **kwargs)


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

    # TODO: may eventually do file format checking
    argparser.add_argument(
        "-if",
        "--input-format",
        required=False,
        help=(
            "Load the provided files as the given type, overriding built-in "
            "filename extension mapping."
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
        "--print-filename",
        action="store_true",
        help="Print filenames along with results",
    )

    argparser.add_argument(
        "--print-source",
        action="store_true",
        help="Dump the source code",
    )

    argparser.add_argument(
        "--print-tree",
        action="store_true",
        help="Dump the source code tree",
    )

    argparser.add_argument(
        "--debug",
        action="store_true",
        help="On failure, still return the results tree",
    )

    argparser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter IPython (or Python) to explore source trees",
    )

    argparser.add_argument(
        "-s",
        "--summary",
        dest="output_summary",
        action="store_true",
        help="Summarize code inputs and outputs",
    )

    argparser.add_argument(
        "--json",
        dest="use_json",
        action="store_true",
        help="Output JSON representation only",
    )

    argparser.add_argument(
        "--no-meta",
        action="store_false",
        dest="include_meta",
        help="Summarize code inputs and outputs",
    )

    argparser.add_argument(
        "--filter",
        nargs="*",
        dest="filter_by_name",
        help="Filter items to parse by name",
    )

    return argparser


def summarize(parsed: list[ParseResult]) -> summary.CodeSummary:
    """Get a code summary instance from one or more ParseResult instances."""
    return summary.CodeSummary.from_parse_results(parsed)


T = TypeVar("T")


def dump_json(
    type_: Type[T],
    obj: T,
    include_meta: bool = True,
    indent: Optional[int] = 2,
) -> str:
    """
    Dump object ``obj`` as type ``type_`` with apischema and serialize to a string.

    Parameters
    ----------
    type_ : Type[T]
        The type of ``obj``.
    obj : T
        The object to serialize.
    include_meta : bool
        Include ``meta`` information in the dump.
    indent : int or None
        Make the JSON output prettier with indentation.

    Returns
    -------
    str
    """
    if apischema is None:
        raise RuntimeError(
            "Optional dependency apischema is required to output a JSON "
            "representation of source code."
        )

    serialized = apischema.serialize(
        type_,
        obj,
        exclude_defaults=True,
        no_copy=True,
    )
    if not include_meta:
        serialized = util.recursively_remove_keys(serialized, {"meta"})
    return json.dumps(serialized, indent=indent)


def main(
    filename: Union[str, pathlib.Path],
    verbose: int = 0,
    debug: bool = False,
    interactive: bool = False,
    output_summary: bool = False,
    use_json: bool = False,
    print_filename: bool = False,
    print_source: bool = False,
    print_tree: bool = False,
    include_meta: bool = True,
    filter_by_name: Optional[list[str]] = None,
    input_format: Optional[str] = None,
) -> dict[str, list[ParseResult]]:
    """
    Parse the given source code/project.
    """
    results_by_filename = {}
    filter_by_name = filter_by_name or []
    filename = pathlib.Path(filename)

    if use_json:
        print_filename = False

        if apischema is None:
            raise RuntimeError(
                "Optional dependency apischema is required to output a JSON "
                "representation of source code."
            )

    print_filename = print_filename or verbose > 1
    print_source = print_source or verbose > 1
    print_tree = print_tree or verbose > 1
    print_tracebacks = verbose > 1

    def get_items():
        for item in load_file_by_name(filename, input_format):
            if filter_by_name:
                if any(flt.lower() in str(filename) for flt in filter_by_name):
                    logger.debug(
                        "Included by filter (filename match): %s (%s)",
                        item.identifier,
                        type(item),
                    )
                elif not any(flt.lower() in item.identifier for flt in filter_by_name):
                    logger.debug("Filtered out: %s (%s)", item.identifier, type(item))
                    continue
                else:
                    logger.debug("Included by filter: %s", item.identifier)
            yield from parse_item(item)

    all_results = []
    try:
        for index, res in enumerate(get_items(), start=1):
            all_results.append(res)
            results_by_filename.setdefault(str(filename), []).append(res)
            if print_filename:
                print(f"[{index}] Parsing {res.filename}: {res.identifier} ({res.item.type})")

            if print_source:
                res.dump_source()

            if print_tree and res.tree is not None:
                print(res.tree.pretty())

            if res.exception is not None:
                tb = getattr(res.exception, "traceback", None)
                if print_tracebacks:
                    print(tb)
                print(
                    f"Failed to parse {res.filename} {res.identifier}: "
                    f"Exception: {type(res.exception).__name__}: {res.exception}"
                )
                if interactive:
                    util.python_debug_session(
                        namespace={"result": res},
                        message=(
                            f"Failed to parse {res.filename} {res.identifier}.\n"
                            f"Exception: {type(res.exception).__name__}: {res.exception}\n"
                            f"{tb}"
                        ),
                    )

            if output_summary:
                # Summary comes at the end
                continue

            if use_json:
                assert apischema is not None

                # We allow for StatementList to be parsed directly here, though
                # it's not acceptable as a top-level item in the grammar
                serialized = dump_json(
                    tf.ExtendedSourceCode,
                    res.transform(),
                    include_meta=include_meta,
                )
                print(serialized)

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt; stopping parsing.")

    if output_summary:
        summarized = summarize(all_results)
        if use_json:
            print(dump_json(summary.CodeSummary, summarized, include_meta=include_meta))
        else:
            print(summarized)

    if not results_by_filename:
        return {}

    results = []
    for _, items in results_by_filename.items():
        results.extend(items)
    failures = [item for item in results if item.exception is not None]

    if interactive:
        util.python_debug_session(
            namespace={
                "results": results,
                "by_filename": results_by_filename,
                "failures": failures,
            },
            message=(
                f"Saw {len(results_by_filename)} files with {len(results)} "
                f"total source code items.\n"
                f"There were {len(failures)} failures.\n"
                f"Results by filename are in ``by_filename``.\n"
                f"All results are also in a list ``results``.\n"
                f"Any failures are included in ``failures``.\n"
            ),
        )

    if failures:
        print("Failed to parse some source code files:")
        for failure in failures:
            header = f"{failure.filename}: {failure.identifier}"
            print(header)
            print("-" * len(header))
            print(f"({type(failure.exception).__name__}) {failure.exception}")
            print()
            # traceback.print_exc()

        if not debug:
            sys.exit(1)

    return results_by_filename
