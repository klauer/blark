import pathlib

MODULE_PATH = pathlib.Path(__file__).parent
del pathlib

GRAMMAR_FILENAME = MODULE_PATH / 'iec.lark'

__all__ = ['GRAMMAR_FILENAME']
