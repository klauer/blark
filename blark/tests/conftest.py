import dataclasses
import functools
import inspect
import pathlib

import lark
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


def stringify_tokens(obj):
    if obj is None:
        return
    if isinstance(obj, list):
        return [stringify_tokens(part) for part in obj]
    if isinstance(obj, tuple):
        return tuple(stringify_tokens(part) for part in obj)
    if isinstance(obj, dict):
        return {
            key: stringify_tokens(value)
            for key, value in obj.items()
        }

    if isinstance(obj, lark.Token):
        return str(obj)

    if dataclasses.is_dataclass(obj):
        for attr, item in inspect.getmembers(obj):
            if attr.startswith("_"):
                continue

            try:
                setattr(obj, attr, stringify_tokens(item))
            except AttributeError:
                ...

    return obj
