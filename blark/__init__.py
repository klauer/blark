import pathlib

from . import _version, plain
from .parse import get_parser, new_parser, parse_project, parse_source_code
from .solution import Project, Solution, TwincatTsProject
from .transform import GrammarTransformer

__version__ = _version.get_versions()["version"]

MODULE_PATH = pathlib.Path(__file__).parent
del pathlib

GRAMMAR_FILENAME = MODULE_PATH / "iec.lark"

plain._register()

__all__ = [
    "GRAMMAR_FILENAME",
    "GrammarTransformer",
    "Project",
    "Solution",
    "TwincatTsProject",
    "get_parser",
    "new_parser",
    "parse_project",
    "parse_source_code",
]
