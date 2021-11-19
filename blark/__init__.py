import pathlib

from . import _version
from .parse import get_parser, parse_project, parse_source_code
from .transform import GrammarTransformer

__version__ = _version.get_versions()["version"]

MODULE_PATH = pathlib.Path(__file__).parent
del pathlib

GRAMMAR_FILENAME = MODULE_PATH / "iec.lark"

__all__ = [
    "GRAMMAR_FILENAME",
    "GrammarTransformer",
    "get_parser",
    "parse_project",
    "parse_source_code",
]
