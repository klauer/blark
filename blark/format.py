"""
`blark format` is a command-line utility to parse and print formatted TwinCAT3
source code files.
"""
import argparse
import pathlib
import sys

import pytmc

from .parse import parse_project, parse_single_file

DESCRIPTION = __doc__


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
            if debug:
                raise
        else:
            print(results[fn])

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
