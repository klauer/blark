from __future__ import annotations

import codecs
import dataclasses
import enum
import hashlib
import os
import pathlib
import re
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TypeVar

import lark
import lxml.etree

from .typing import AnyPath, DeclarationOrImplementation, Self

RE_LEADING_WHITESPACE = re.compile("^[ \t]+", re.MULTILINE)
NEWLINES = "\n\r"
SINGLE_COMMENT = "//"
OPEN_COMMENT = "(*"
CLOSE_COMMENT = "*)"
OPEN_PRAGMA = "{"
CLOSE_PRAGMA = "}"


class SourceType(enum.Enum):
    general = enum.auto()
    action = enum.auto()
    function = enum.auto()
    function_block = enum.auto()
    interface = enum.auto()
    method = enum.auto()
    program = enum.auto()
    property = enum.auto()
    property_get = enum.auto()
    property_set = enum.auto()
    dut = enum.auto()
    statement_list = enum.auto()
    var_global = enum.auto()

    def __str__(self) -> str:
        return self.name

    def get_grammar_rule(self) -> str:
        return {
            SourceType.action: "statement_list",
            SourceType.function: "function_declaration",
            SourceType.function_block: "function_block_type_declaration",
            SourceType.general: "iec_source",
            SourceType.interface: "interface_declaration",
            SourceType.method: "function_block_method_declaration",
            SourceType.program: "program_declaration",
            SourceType.property: "function_block_property_declaration",
            SourceType.property_get: "function_block_property_declaration",
            SourceType.property_set: "function_block_property_declaration",
            SourceType.statement_list: "statement_list",
            SourceType.dut: "data_type_declaration",
            # NOTE: multiple definitions can be present in GVLs:
            SourceType.var_global: "iec_source",
        }[self]

    def get_implicit_block_end(self) -> str:
        return {
            SourceType.action: "",
            SourceType.function: "END_FUNCTION",
            SourceType.function_block: "END_FUNCTION_BLOCK",
            SourceType.general: "",
            SourceType.interface: "END_INTERFACE",
            SourceType.method: "END_METHOD",
            SourceType.program: "END_PROGRAM",
            SourceType.property: "END_PROPERTY",
            SourceType.property_get: "",
            SourceType.property_set: "",
            SourceType.statement_list: "",
            SourceType.dut: "",
            SourceType.var_global: "",
        }[self]


@dataclasses.dataclass
class Identifier:
    """
    A blark convention for giving portions of code unique names.

    Examples of valid identifiers include:

    * FB_Name/declaration
    * FB_Name/implementation
    * FB_Name.Action/declaration
    * FB_Name.Action/implementation
    * FB_Name.Property.get/implementation
    * FB_Name.Property.set/implementation

    Attributes
    ----------
    parts : list of str
        Parts of the name, split by the "." character.
    decl_impl : "declaration" or "implementation"
        The final "/portion", indicating whether the code section is describing
        the declaration portion or the implementation portion.
    """
    parts: List[str]
    decl_impl: Optional[DeclarationOrImplementation] = None

    @property
    def dotted_name(self) -> str:
        return ".".join(self.parts)

    def to_string(self) -> str:
        parts = ".".join(self.parts)
        if self.decl_impl:
            return f"{parts}/{self.decl_impl}"
        return parts

    @classmethod
    def from_string(cls: type[Self], value: str) -> Self:
        if "/" in value:
            identifier, decl_impl = value.split("/")
            assert decl_impl in {"declaration", "implementation", None}
            return cls(
                parts=identifier.split("."),
                decl_impl=decl_impl,
            )
        return cls(
            parts=value.split("."),
            decl_impl=None,
        )


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
    from .input import load_file_by_name
    result = []
    for item in load_file_by_name(fn):
        code, _ = item.get_code_and_line_map()
        result.append(code)

    return "\n\n".join(result)


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

    debug_namespace = {"blark": blark}
    debug_namespace.update(
        **{k: v for k, v in namespace.items() if not k.startswith("__")}
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


def find_pou_type_and_identifier(code: str) -> tuple[Optional[SourceType], Optional[str]]:
    types = {source.name for source in SourceType}
    clean_code = remove_all_comments(code)
    for line in clean_code.splitlines():
        parts = line.lstrip().split()
        if parts and parts[0].lower() in types:
            source_type = SourceType[parts[0].lower()]
            identifier = None
            if source_type != SourceType.var_global:
                for identifier in parts[1:]:
                    if identifier.lower() not in {
                        "abstract",
                        "public",
                        "private",
                        "protected",
                        "internal",
                        "final",
                    }:
                        break
            return source_type, identifier
    return None, None


def remove_all_comments(text: str, *, replace_char: str = " ") -> str:
    """
    Remove all comments and replace them with the provided character.
    """
    # TODO review the logic here! it's Friday after 5PM
    multiline_comments = []
    in_single_comment = False
    in_single_quote = False
    in_double_quote = False
    pragma_state = []
    skip = 0

    def get_characters() -> Generator[Tuple[int, int, str, str], None, None]:
        """Yield line information and characters."""
        for lineno, line in enumerate(text.splitlines()):
            colno = 0
            for colno, (this_ch, next_ch) in enumerate(zip(line, line[1:] + "\n")):
                yield lineno, colno, this_ch, next_ch
            yield lineno, colno, "\n", ""

    result = []
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
                pragma_state.pop(-1)
                continue

            if pragma_state:
                continue

            if pair == OPEN_COMMENT:
                multiline_comments.append((lineno, colno))
                skip = 1
                continue
            if pair == CLOSE_COMMENT:
                multiline_comments.pop(-1)
                skip = 1
                continue
            if pair == SINGLE_COMMENT:
                in_single_comment = True
                continue

        if not multiline_comments and not in_single_comment:
            if pair == "$'" and in_single_quote:
                # This is an escape for single quotes
                skip = 1
                result.append(pair)
            elif pair == '$"' and in_double_quote:
                # This is an escape for double quotes
                skip = 1
                result.append(pair)
            elif this_ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                result.append(this_ch)
            elif this_ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                result.append(this_ch)
            elif pair == SINGLE_COMMENT:
                in_single_comment = True
            else:
                result.append(this_ch)

    if multiline_comments or in_single_quote or in_double_quote:
        # Syntax error in source? Return the original and let lark fail
        return text

    return "".join(result)


def find_and_clean_comments(
    text: str,
    *,
    replace_char: str = " ",
    line_map: Optional[dict[int, int]] = None,
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

    def get_token(
        start_line: int, start_col: int, end_line: int, end_col: int
    ) -> lark.Token:
        if start_line != end_line:
            block = "\n".join(
                (
                    original_lines[start_line][start_col:],
                    *original_lines[start_line + 1: end_line],
                    original_lines[end_line][: end_col + 1],
                )
            )
        else:
            block = original_lines[start_line][start_col: end_col + 1]

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
                    "SINGLE_LINE_COMMENT": "",  # this would be a bug
                    "MULTI_LINE_COMMENT": "    ",
                    "PRAGMA": "    ",
                }[type_],
            )

        token = lark.Token(
            type_,
            block,
            line=start_line + 1,
            end_line=end_line + 1,
            column=start_col + 1,
            end_column=end_col + 1,
        )
        if line_map is not None:
            token.line = line_map[start_line + 1]
            token.end_line = line_map[end_line + 1]
            # token.line = line_map.get(start_line + 1, start_line + 1)
            # token.end_line = line_map.get(end_line + 1, end_line + 1)
        return token

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
                    get_token(lineno, colno, lineno, len(lines[lineno]))
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


def get_file_sha256(filename: AnyPath) -> str:
    """Hash a file's contents with the SHA-256 algorithm."""
    with open(filename, "rb") as fp:
        return hashlib.sha256(fp.read()).hexdigest()


def fix_case_insensitive_path(path: AnyPath) -> pathlib.Path:
    """
    Match a path in a case-insensitive manner.

    Required on Linux to find files in a case-insensitive way. Not required on
    OSX/Windows, but platform checks are not done here.

    Parameters
    ----------
    path : pathlib.Path or str
        The case-insensitive path

    Returns
    -------
    path : pathlib.Path
        The case-corrected path.

    Raises
    ------
    FileNotFoundError
        When the file can't be found
    """
    path = pathlib.Path(path).expanduser().resolve()
    if path.exists():
        return path.resolve()

    new_path = pathlib.Path(path.parts[0])
    for part in path.parts[1:]:
        if not (new_path / part).exists():
            all_files = {fn.name.lower(): fn.name for fn in new_path.iterdir()}
            try:
                part = all_files[part.lower()]
            except KeyError:
                raise FileNotFoundError(
                    f"Path does not exist: {path}\n{new_path}{os.pathsep}{part} missing"
                ) from None
        new_path = new_path / part
    return new_path.resolve()


def try_paths(paths: List[AnyPath]) -> Optional[pathlib.Path]:
    for path in paths:
        try:
            return fix_case_insensitive_path(path)
        except FileNotFoundError:
            pass

    options = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the possible files were found:\n{options}")


_T_Lark = TypeVar("_T_Lark", lark.Tree, lark.Token)


def rebuild_lark_tree_with_line_map(
    item: _T_Lark, code_line_to_file_line: dict[int, int]
) -> _T_Lark:
    """Rebuild a given lark tree, adjusting line numbers to match up with the source."""
    if isinstance(item, lark.Token):
        if item.line is not None:
            item.line = code_line_to_file_line.get(item.line, item.line)
        if item.end_line is not None:
            item.end_line = code_line_to_file_line.get(item.end_line, item.end_line)
        return item

    if not isinstance(item, lark.Tree):
        raise NotImplementedError(f"Type: {item.__class__.__name__}")

    try:
        meta = item.meta
    except AttributeError:
        meta = None
    else:
        if not meta.empty:
            meta.line = code_line_to_file_line.get(meta.line, meta.line)
            meta.end_line = code_line_to_file_line.get(meta.end_line, meta.end_line)

    return lark.Tree(
        item.data,
        children=[
            None
            if child is None
            else rebuild_lark_tree_with_line_map(child, code_line_to_file_line)
            for child in item.children
        ],
        meta=meta,
    )


def tree_to_xml_source(
    tree: lxml.etree.Element,
    encoding: str = "utf-8",
    delimiter: str = "\r\n",
    xml_header: str = '<?xml version="1.0" encoding="{encoding}"?>',
    indent: str = "  ",
    include_utf8_sig: bool = True,
) -> bytes:
    """Return the contents to write for the given XML tree."""
    # NOTE: we avoid lxml.etree.tostring(xml_declaration=True) as we want
    # to write a declaration that matches what TwinCAT writes. It uses double
    # quotes instead of single quotes.
    delim_bytes = delimiter.encode(encoding)
    header_bytes = xml_header.format(encoding=encoding).encode(encoding)
    lxml.etree.indent(tree, space=indent)
    if encoding.startswith("utf-8") and include_utf8_sig:
        # Additionally, TwinCAT includes a utf-8 byte order marker (BOM).
        # Let's include that or our formatted output will differ.
        header_bytes = codecs.BOM_UTF8 + header_bytes

    source = header_bytes + delim_bytes + lxml.etree.tostring(
        tree,
        pretty_print=True,
        encoding=encoding,
    )

    if delim_bytes == b"\n":
        # This is what lxml gives us
        return source

    source_lines = source.split(b"\n")
    return delim_bytes.join(source_lines)


def recursively_remove_keys(obj, keys: Set[str]) -> Any:
    """Remove the provided keys from the JSON object."""
    if isinstance(obj, dict):
        return {key: recursively_remove_keys(value, keys) for key, value in obj.items()
                if key not in keys}
    if isinstance(obj, (list, tuple)):
        return [recursively_remove_keys(value, keys) for value in obj]
    return obj
