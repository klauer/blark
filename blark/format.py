"""
`blark format` is a command-line utility to parse and print formatted TwinCAT3
source code files.
"""
import argparse

from .parse import main as parse_main
from .typing import SupportsCustomSave, SupportsRewrite, SupportsWrite
from .util import AnyPath

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

    # argparser.add_argument(
    #     "--output-format",
    #     type=str,
    #     help="Output file format"
    # )

    argparser.add_argument(
        "--in-place",
        action="store_true",
        help="Write formatted contents back to the file"
    )

    return argparser


def main(
    filename: AnyPath,
    verbose: int = 0,
    debug: bool = False,
    interactive: bool = False,
    in_place: bool = False,
):
    result_by_filename = parse_main(
        filename, verbose=verbose, debug=debug, interactive=interactive
    )

    for filename, results in result_by_filename.items():
        for res in results:
            print()
            item = res.item
            if item is None:
                continue
            user = item.user

            if verbose > 1:
                res.dump_source()

            formatted_code = str(res.transform())

            if isinstance(user, SupportsRewrite):
                # print(user.to_file_contents())
                user.rewrite_code(res.identifier, formatted_code)
                # print(user.to_file_contents())

            if isinstance(user, SupportsCustomSave):
                if in_place:
                    user.save_to(filename)
                else:
                    print(formatted_code)
            elif isinstance(user, SupportsWrite):
                contents = user.to_file_contents()
                if in_place:
                    mode = "wb" if isinstance(contents, bytes) else "wt"
                    with open(filename, mode) as fp:
                        fp.write(contents)
                else:
                    print(contents.decode())
            else:
                print(formatted_code)

    return result_by_filename
