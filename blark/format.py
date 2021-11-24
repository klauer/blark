"""
`blark format` is a command-line utility to parse and print formatted TwinCAT3
source code files.
"""
import argparse
import pathlib
import sys
import traceback

from .parse import parse
from .util import AnyPath, python_debug_session

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


def main(
    filename: AnyPath,
    verbose: int = 0,
    debug: bool = False,
    interactive: bool = False,
):
    result_by_filename = {}
    failures = []
    print_filenames = sys.stdout if verbose > 0 else None
    filename = pathlib.Path(filename)

    for fn, result in parse(filename):
        if print_filenames:
            print(f"* Loading {fn}")
        result_by_filename[fn] = result
        if isinstance(result, Exception):
            failures.append((fn, result))
            if interactive:
                python_debug_session(
                    namespace={"fn": fn, "result": result},
                    message=(
                        f"Failed to parse {fn}. {type(result).__name__}: {result}\n"
                        f"{result.traceback}"
                    ),
                )
            elif verbose > 1:
                print(result.traceback)
        else:
            print(result)

    if not result_by_filename:
        return {}

    if interactive:
        if len(result_by_filename) > 1:
            python_debug_session(
                namespace={"fn": filename, "results": result_by_filename},
                message=(
                    "Parsed all files successfully: {list(result_by_filename)}\n"
                    "Access all results by filename in the variable ``results``"
                )
            )
        else:
            ((filename, result),) = list(result_by_filename.items())
            python_debug_session(
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
