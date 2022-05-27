import os
from typing import Any, Dict

import pytest
from pytest import param

from ..format import main as format_main
from ..parse import main as parse_main
from . import conftest

parse_filenames = (
    conftest.twincat_pou_filenames[:5] + conftest.structured_text_filenames[:5]
)


@pytest.fixture(params=parse_filenames)
def input_filename(request):
    if not os.path.exists(request.param):
        pytest.skip(f"File missing: {request.param}")
    return request.param


@pytest.mark.parametrize(
    "kwargs",
    [
        param(
            dict(),
            id="defaults"
        ),
        param(
            dict(use_json=True),
            id="json",
        ),
        param(
            dict(verbose=3, summary=True, debug=True),
            id="verbose",
        ),
    ]
)
def test_parse_cli(input_filename: str, kwargs: Dict[str, Any]):
    parse_main(input_filename, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        param(
            dict(),
            id="defaults"
        ),
        param(
            dict(verbose=3, debug=True),
            id="verbose",
        ),
    ]
)
def test_format_cli(input_filename: str, kwargs: Dict[str, Any]):
    format_main(input_filename, **kwargs)
