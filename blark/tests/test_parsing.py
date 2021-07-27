import pathlib

import lark
import pytest

import blark
import blark.util

TEST_PATH = pathlib.Path(__file__).parent


@pytest.fixture(scope='session')
def iec_parser():
    with open(blark.GRAMMAR_FILENAME, 'rt') as f:
        lark_grammar = f.read()

    return lark.Lark(lark_grammar, parser='earley')


pous = list(str(path) for path in TEST_PATH.glob('**/*.TcPOU'))
additional_pous = TEST_PATH / 'additional_pous.txt'

if additional_pous.exists():
    pous += open(additional_pous, 'rt').read().splitlines()


@pytest.fixture(params=pous)
def pou_filename(request):
    return request.param


@pytest.fixture
def pou_source(pou_filename):
    if not pathlib.Path(pou_filename).exists():
        pytest.skip(f'File does not exist: {pou_filename}')

    return blark.util.get_source_code(pou_filename)


def test_parsing(iec_parser, pou_filename, pou_source):
    try:
        tree = iec_parser.parse(pou_source)
    except Exception:
        print(f'Failed to parse {pou_filename}:')
        print('-------------------------------')
        print(pou_source)
        print('-------------------------------')
        print(f'[Failure] End of {pou_filename}')
        raise
    else:
        print(f'Successfully parsed {pou_filename}:')
        print('-------------------------------')
        print(pou_source)
        print('-------------------------------')
        print(tree.pretty())
        print('-------------------------------')
        print(f'[Success] End of {pou_filename}')


def pytest_html_results_table_row(report, cells):
    # pytest results using pytest-html; show only failures for now:
    if report.passed:
        del cells[:]
