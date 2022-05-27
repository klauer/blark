"""
`blark` is the top-level command for accessing various subcommands.

Try::

"""

import argparse
import importlib
import logging

import blark

DESCRIPTION = __doc__


MODULES = ("parse", "format")


def _try_import(module):
    relative_module = f".{module}"
    return importlib.import_module(relative_module, "blark")


def _build_commands():
    global DESCRIPTION
    result = {}
    unavailable = []

    for module in sorted(MODULES):
        try:
            mod = _try_import(module)
        except ImportError as ex:
            unavailable.append((module, ex))
        else:
            result[module] = (mod.build_arg_parser, mod.main)
            DESCRIPTION += f"\n    $ blark {module} --help"

    if unavailable:
        DESCRIPTION += "\n\n"

        for module, ex in unavailable:
            DESCRIPTION += (
                f"WARNING: blark {module!r} is unavailable due to:"
                f"\n\t{ex.__class__.__name__}: {ex}"
            )

    return result


COMMANDS = _build_commands()


def main():
    # Console entry point in setup.py
    top_parser = argparse.ArgumentParser(
        prog="blark",
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    top_parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=blark.__version__,
        help="Show the blark version number and exit.",
    )

    top_parser.add_argument(
        "--log",
        "-l",
        dest="log_level",
        default="INFO",
        type=str,
        help="Python logging level (e.g. DEBUG, INFO, WARNING)",
    )

    subparsers = top_parser.add_subparsers(help="Possible subcommands")
    for command_name, (build_func, main) in COMMANDS.items():
        sub = subparsers.add_parser(command_name)
        build_func(sub)
        sub.set_defaults(func=main)

    args = top_parser.parse_args()
    kwargs = vars(args)
    log_level = kwargs.pop("log_level")

    logger = logging.getLogger("blark")
    logger.setLevel(log_level)
    logging.basicConfig()

    if hasattr(args, "func"):
        func = kwargs.pop("func")
        logger.debug("%s(**%r)", func.__name__, kwargs)
        func(**kwargs)
    else:
        top_parser.print_help()
