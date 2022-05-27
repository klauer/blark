import os
import sys
from typing import Any, Dict, List

import pytest
from pytest import param

from ..format import main as format_main
from ..main import main as blark_main
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


@pytest.mark.parametrize(
    "args",
    [
        param(
            ["--help"],
            id="top-help",
        ),
        param(
            ["parse", "--help"],
            id="parse-help",
        ),
        param(
            ["format", "--help"],
            id="format-help",
        ),
    ]
)
def test_blark_main_help(monkeypatch, args: List[str]):
    monkeypatch.setattr(sys, "argv", ["blark", *args])
    try:
        blark_main()
    except SystemExit as ex:
        assert ex.code == 0


@pytest.mark.parametrize(
    "args",
    [
        param(
            ["parse", "filename"],
            id="basic-parse",
        ),
        param(
            ["parse", "-vvvs", "--debug", "filename"],
            id="parse-verbose",
        ),
        param(
            ["parse", "--json", "filename"],
            id="parse-json",
        ),
        param(
            ["format", "filename"],
            id="format-basic",
        ),
        param(
            ["format", "--debug", "filename"],
            id="format-debug",
        ),
    ]
)
def test_blark_main(monkeypatch, input_filename: str, args: List[str]):
    def replace_filename(arg: str) -> str:
        if arg == "filename":
            return input_filename
        return arg

    args = [replace_filename(arg) for arg in args]

    monkeypatch.setattr(sys, "argv", ["blark", *args])
    try:
        blark_main()
    except SystemExit as ex:
        assert ex.code == 0
