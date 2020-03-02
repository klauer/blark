import pathlib
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

MODULE_PATH = pathlib.Path(__file__).parent
del pathlib

GRAMMAR_FILENAME = MODULE_PATH / 'iec.lark'

__all__ = ['GRAMMAR_FILENAME']
