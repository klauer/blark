"""
`blark format` is a command-line utility to parse and print formatted TwinCAT3
source code files.
"""
import argparse
import logging
import pathlib
import sys
from typing import Any, List, Optional, Tuple, Union

from blark import output

from . import transform as tf
from .output import OutputBlock
from .parse import ParseResult
from .parse import main as parse_main
from .util import AnyPath

DESCRIPTION = __doc__

logger = logging.getLogger(__name__)


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
        "-if",
        "--input-format",
        type=str,
        help="Output file format, if not the same as the input format",
    )
    argparser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity, up to -vvv",
    )

    argparser.add_argument(
        "--indent",
        default="    ",
        help="Single indent to reformat with",
    )

    argparser.add_argument(
        "--debug",
        action="store_true",
        help="On failure, still return the results tree",
    )

    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )

    argparser.add_argument(
        "--write-to",
        type=str,
        help="Write formatted contents to this path",
    )

    argparser.add_argument(
        "-of",
        "--output-format",
        type=str,
        help="Output file format, if not the same as the input format",
    )
    return argparser


def reformat_code(
    code: tf.SourceCode,
) -> str:
    """Reformat the code with the provided settings."""
    # For anyone following at home, this is not a "good" formatting
    # of the output - just a consistent output formatting.
    # Opinionated, stylized output remains a TODO.
    return str(code)


def dump_source_to_console(source: Union[bytes, str], encoding: str = "utf-8") -> None:
    """
    Output the given source to the console.

    Parameters
    ----------
    source : bytes or str
        The source code.
    encoding : str
        Encoding to use for byte strings.
    """
    if isinstance(source, bytes):
        print(source.decode(encoding, errors="replace"))
    else:
        print(source)


def write_source_to_file(
    filename: pathlib.Path,
    source: Union[bytes, str],
    encoding: str = "utf-8",
    overwrite: bool = False,
) -> None:
    """
    Write source code to the given file.

    Parameters
    ----------
    filename : pathlib.Path
        The filename to write to.
    source : bytes or str
        The source code.
    encoding : str
        The encoding to use when writing the file.
    overwrite : bool
        Overwrite a file, if it exists.  Otherwise, raise ValueError.
    """
    if filename.exists() and not overwrite:
        raise ValueError(
            f"File would be overwritten: {filename} "
            f"if this is OK use --overwrite"
        )

    if isinstance(source, str):
        # NOTE: if this is undesirable, make your output handler
        # return bytes instead of a string
        source = source.encode(encoding)

    with open(filename, "wb") as fp:
        fp.write(source)


def determine_output_filename(
    input_filename: pathlib.Path,
    write_to: Optional[pathlib.Path],
) -> pathlib.Path:
    """
    Get an output filename based on the input filename and destination path.

    Parameters
    ----------
    input_filename : pathlib.Path
        The file the source code comes from.
    write_to : Optional[pathlib.Path]
        The destination path to write to.

    Returns
    -------
    pathlib.Path
        The output filename to write to.
    """
    if write_to:
        output_filename = pathlib.Path(write_to)
        if output_filename.is_dir():
            return output_filename / input_filename.name
        return output_filename

    return input_filename


def get_reformatted_code_blocks(
    results: List[ParseResult],
    user: Optional[Any] = None,
    filename: Optional[pathlib.Path] = None,
    raise_on_error: bool = True,
) -> List[OutputBlock]:
    """
    For each parsed code block, generate an OutputBlock for writing to disk.

    Parameters
    ----------
    results : List[ParseResult]
        The parsed source code.
    user : Any, optional
        The loader used for parsing the above (may be a
        `blark.plain.PlainFileLoader` or a `blark.solution.TcPOU`, for example)
    filename : pathlib.Path, optional
        The filename associated with the source code parts.
    raise_on_error : bool
        In the event of a reformatting error, raise immediately.  If False,
        the original (not reformatted) source code will be used as-is.

    Returns
    -------
    List[OutputBlock]
        [TODO:description]
    """
    blocks: List[OutputBlock] = []

    logger.debug(
        "Reformatting %d parse results from file %s by way of loader %s",
        len(results),
        filename,
        user,
    )
    for res in results:
        try:
            formatted = reformat_code(res.transform())
        except Exception:
            logger.exception(
                "Failed to reformat code block for identifier %s",
                res.identifier,
            )
            if raise_on_error:
                raise
            formatted = res.source_code

        blocks.append(
            OutputBlock(
                code=formatted,
                metadata={},  # what might go here?
                origin=res,
            )
        )
    return blocks


def main(
    filename: AnyPath,
    verbose: int = 0,
    debug: bool = False,
    interactive: bool = False,
    indent: str = "    ",
    write_to: Optional[AnyPath] = None,
    overwrite: bool = False,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
):
    if write_to is not None:
        write_to = pathlib.Path(write_to)

    result_by_filename = parse_main(
        filename,
        verbose=verbose,  # deprecated
        debug=debug,
        interactive=interactive,
    )

    settings = tf.FormatSettings(
        indent=indent,
    )
    tf.configure_formatting(settings)

    def group_by_user_loader() -> List[Tuple[Any, pathlib.Path, List[ParseResult]]]:
        """
        Group all ParseResults by their loader (or "user" defined wrapper.)

        This grouping ensures that things like POUs (which may contain multiple
        actions, methods, and properties stored in different parts of a source
        file) have their rewrite stage happen in two steps: (1) reformat and
        then rewrite all disjoint parts of the source and (2) write the final
        results to the user-specified destination.
        """
        by_loader_obj = []
        last_user = object()
        last_filename = object()
        current_group = []
        for _, results in result_by_filename.items():
            for res in results:
                user = res.item.user
                if res.item.user is not last_user or res.filename != last_filename:
                    current_group = [res]
                    by_loader_obj.append((user, res.filename, current_group))
                    last_user = user
                    last_filename = res.filename
                else:
                    current_group.append(res)

        return by_loader_obj

    for user, filename, results in group_by_user_loader():
        blocks = get_reformatted_code_blocks(results, user=user, filename=filename)
        if not blocks:
            continue

        # Default to 'output format' if specified
        # Fall back to writing to the user-provided input format
        # Finally try to detect the input format to determine the output format
        output_handler = output.get_handler_by_name(
           output_format or input_format or filename.suffix
        )
        logger.debug(
            "Chose output handler %s based on %r -> %r -> %r (%s)",
            output_handler,
            output_format,
            input_format,
            filename.suffix,
            filename,
        )

        # Possibilities:
        # 1. Save to same file in same format (--overwrite)
        # 2. Save to different file/directory in same format (--write-to)
        # 3. Save to different file in different format (--output-format, --write-to)
        # 4. Save to same file in different format (--output-format xyz, --overwrite)

        # For now, we'll only *really* support cases (1) and (2)
        # Exceptions:
        # - Allow anything->plain conversion because it's easy.

        try:
            output_contents = output_handler(user, filename, blocks)
        except Exception:
            logger.exception(
                "Failed to transform output for handler %s",
                output_format,
            )
            sys.exit(1)

        if not write_to and not overwrite:
            dump_source_to_console(output_contents)
            continue

        output_filename = determine_output_filename(filename, write_to)
        write_source_to_file(output_filename, output_contents, overwrite=overwrite)

    return result_by_filename
