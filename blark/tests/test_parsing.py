import pathlib

import pytest

from ..parse import parse_single_file

TEST_PATH = pathlib.Path(__file__).parent


pous = list(str(path) for path in TEST_PATH.glob('**/*.TcPOU'))
additional_pous = TEST_PATH / 'additional_pous.txt'

if additional_pous.exists():
    pous += open(additional_pous, 'rt').read().splitlines()


@pytest.fixture(params=pous)
def pou_filename(request):
    return request.param


def test_parsing(pou_filename):
    try:
        parse_single_file(pou_filename, verbose=2)
    except FileNotFoundError:
        pytest.skip(f"Missing file: {pou_filename}")


def pytest_html_results_table_row(report, cells):
    # pytest results using pytest-html; show only failures for now:
    if report.passed:
        del cells[:]
