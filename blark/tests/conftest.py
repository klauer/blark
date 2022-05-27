import dataclasses
import functools
import json
import pathlib

import pytest

from ..parse import new_parser

TEST_PATH = pathlib.Path(__file__).parent

try:
    import apischema
except ImportError:
    # apischema is optional for serialization testing
    apischema = None

APISCHEMA_SKIP = apischema is None


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


def roundtrip_serialization(obj):
    """
    Round-trip a dataclass object with the serialization library.

    Requires apischema and APISCHEMA_SKIP to be False.

    Checks:
    * ``obj`` can be serialized to JSON
    * Serialized JSON can be deserialized back into an equivalent ``obj``
    * Deserialized object has the same source code representation
    """
    if obj is None or apischema is None or APISCHEMA_SKIP:
        return

    try:
        serialized = apischema.serialize(obj)
    except Exception:
        print(json.dumps(dataclasses.asdict(obj), indent=2))
        raise

    print(f"Serialized {type(obj)} to:")
    print(json.dumps(serialized, indent=2))
    deserialized = apischema.deserialize(type(obj), serialized)

    print(f"Deserialized {type(obj)} back to:")
    print(repr(deserialized))
    print("Or:")
    print(deserialized)

    assert str(obj) == str(deserialized), \
        "Deserialized object does not produce identical source code"

    return serialized, deserialized
