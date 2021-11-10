"""
`blark parse` is a command-line utility to parse TwinCAT3 source code
files in conjunction with pytmc.
"""
import argparse
import pathlib
import re
import sys

import lark
import pytmc

import blark

from .transform import GrammarTransformer
from .util import get_source_code

DESCRIPTION = __doc__
RE_COMMENT = re.compile(r"(//.*$|\(\*.*?\*\))", re.MULTILINE | re.DOTALL)
RE_PRAGMA = re.compile(r"{[^}]*?}", re.MULTILINE | re.DOTALL)


_PARSER = None


def get_parser() -> lark.Lark:
    "Get the global lark.Lark parser for IEC61131-3 code"
    global _PARSER

    if _PARSER is None:
        _PARSER = lark.Lark.open_from_package(
            "blark",
            blark.GRAMMAR_FILENAME.name,
            parser="earley",
            maybe_placeholders=True,
        )
    return _PARSER


def replace_comments(text: str, *, replace_char: str = " ") -> str:
    """
    Clean nested multiline comments from ``text``.

    For a nested comment like ``"(* (* abc *) *)"``, the inner comment markers
    would be replaced with ``replace_char``, resulting in the return value
    ``"(*    abc    *)"``.
    """
    result = list(text)
    in_multiline_comment = 0
    in_single_comment = False
    in_single_quote = False
    in_double_quote = False
    skip = 0
    NEWLINES = "\n\r"
    SINGLE_COMMENT = "//"
    OPEN_COMMENT = "(*"
    CLOSE_COMMENT = "*)"
    for idx, (this_ch, next_ch) in enumerate(zip(text, text[1:] + " ")):
        if skip:
            skip -= 1
            continue

        if in_single_comment:
            in_single_comment = this_ch not in NEWLINES
            continue

        pair = this_ch + next_ch
        if not in_single_quote and not in_double_quote:
            if pair == OPEN_COMMENT:
                in_multiline_comment += 1
                skip = 1
                if in_multiline_comment > 1:
                    # Nested multi-line comment
                    result[idx] = replace_char
                    result[idx + 1] = replace_char
                continue
            if pair == CLOSE_COMMENT:
                in_multiline_comment -= 1
                if in_multiline_comment > 0:
                    # Nested multi-line comment
                    result[idx] = replace_char
                    result[idx + 1] = replace_char
                skip = 1
                continue
            if pair == SINGLE_COMMENT:
                in_single_comment = True
                continue

        if not in_multiline_comment and not in_single_comment:
            if pair == "$'" and in_single_quote:
                # This is an escape for single quotes
                skip = 1
                continue
            elif pair == '$"' and in_double_quote:
                # This is an escape for double quotes
                skip = 1
                continue
            elif this_ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif this_ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif pair == SINGLE_COMMENT:
                in_single_comment = 1

        # if in_multiline_comment > 0 and this_ch not in NEWLINES:
        #     result[idx] = replace_char

    if in_multiline_comment or in_single_quote or in_double_quote:
        # Syntax error in source? Return the original and let lark fail
        return text

    return "".join(result)


# TODO improve grammar to make this step not required :(
DEFAULT_PREPROCESSORS = [replace_comments]
_DEFAULT_PREPROCESSORS = object()


def parse_source_code(
    source_code: str,
    *,
    verbose: int = 0,
    fn: str = "unknown",
    preprocessors=_DEFAULT_PREPROCESSORS
):
    "Parse source code with the parser"
    if preprocessors is _DEFAULT_PREPROCESSORS:
        preprocessors = DEFAULT_PREPROCESSORS

    processed_source = source_code
    for preprocessor in preprocessors:
        processed_source = preprocessor(processed_source)

    try:
        tree = get_parser().parse(processed_source)
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

    # This is some WIP comment + declaration matching
    # pragmas = RE_PRAGMA.findall(source_code)
    # line_numbers = _build_map_of_offset_to_line_number(source_code)
    # comments = list(RE_COMMENT.finditer(source_code))

    # decl, = list(tree.find_data('data_type_declaration'))
    # element1 = [c for c in decl.children][0]
    #
    # print('elem1', element1)
    # # comment 1 at line 9:
    # print(line_numbers[comments[1].end()])
    # # matches up with definition on line 10:
    # print(element1.children[0].children[0].line)
    return GrammarTransformer().transform(tree)


def _build_map_of_offset_to_line_number(source):
    """
    For a multiline source file, return {character_pos: line}
    """
    start_index = 0
    index_to_line_number = {}
    # A slow and bad algorithm, but only to be used in parsing declarations
    # which are rather small
    for line_number, line in enumerate(source.splitlines(), 1):
        for index in range(start_index, start_index + len(line) + 1):
            index_to_line_number[index] = line_number
        start_index += len(line) + 1
    return index_to_line_number


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

            # if '<?xml ' in source_code.splitlines()[0]:
            #     print('found xml')
            #     if name in plc.gvl_by_name or name in plc.dut_by_name:
            #         # TODO pytmc
            #         source_code = source_item.declaration
            #     else:
            #         print('* TODO?', name, source_code)
            #         continue

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

    return argparser


def main(filename, verbose=0, debug=False):
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
        except Exception:
            success = False

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
