import dataclasses
import functools
import json
import os
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

twincat_pou_filenames = list(str(path) for path in TEST_PATH.glob("**/*.TcPOU"))
additional_pous = TEST_PATH / "additional_pous.txt"

if additional_pous.exists():
    twincat_pou_filenames += open(additional_pous, "rt").read().splitlines()

structured_text_filenames = list(str(path) for path in TEST_PATH.glob("**/*.st"))


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


def check_serialization(
    obj, deserialize: bool = True, require_same_source: bool = True
):
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
        serialized = apischema.serialize(
            obj,
            exclude_defaults=True,
            no_copy=True,
        )
    except Exception:
        print(json.dumps(dataclasses.asdict(obj), indent=2))
        raise

    print(f"Serialized {type(obj)} to:")
    print(json.dumps(serialized, indent=2))
    print()

    if not deserialize:
        return serialized, None

    deserialized = apischema.deserialize(type(obj), serialized, no_copy=True)

    print(f"Deserialized {type(obj)} back to:")
    print(repr(deserialized))
    print("Or:")
    print(deserialized)

    if require_same_source:
        assert str(obj) == str(deserialized), \
            "Deserialized object does not produce identical source code"

    return serialized, deserialized


@pytest.fixture(params=twincat_pou_filenames)
def twincat_pou_filename(request):
    if not os.path.exists(request.param):
        pytest.skip(f"File missing: {request.param}")
    return request.param


@pytest.fixture(params=structured_text_filenames)
def source_filename(request):
    if not os.path.exists(request.param):
        pytest.skip(f"File missing: {request.param}")
    return request.param
