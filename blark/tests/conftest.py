import functools
import pathlib

import pytest

from ..parse import new_parser

TEST_PATH = pathlib.Path(__file__).parent


@functools.lru_cache(maxsize=100)
def get_grammar(*, start=None, **kwargs):
    return new_parser(
        import_paths=[TEST_PATH.parent],
        start=start or ["start"],
        **kwargs
    )


@pytest.fixture(scope="module")
def grammar():
    return get_grammar()
