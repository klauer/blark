"""
`blark parse` is a command-line utility to parse TwinCAT3 source code
files in conjunction with pytmc.
"""
import argparse
import pathlib
import re
import sys

import lark
import pytmc

import blark

from .util import get_source_code

DESCRIPTION = __doc__
RE_COMMENT = re.compile(r'(//.*$|\(\*.*?\*\))', re.MULTILINE | re.DOTALL)
RE_PRAGMA = re.compile(r'{[^}]*?}', re.MULTILINE | re.DOTALL)


with open(blark.GRAMMAR_FILENAME, 'rt') as f:
    lark_grammar = f.read()


_PARSER = None


def get_parser():
    'Get the global lark.Lark parser for IEC61131-3 code'
    global _PARSER

    if _PARSER is None:
        _PARSER = lark.Lark(lark_grammar, parser='earley')
    return _PARSER


def replace_comments(text, *, replace_char=' '):
    'Remove (potentially nested) multiline comments from `text`'
    result = []
    in_comment = 0
    skip = 0
    OPEN_COMMENT = ('(', '*')
    CLOSE_COMMENT = ('*', ')')
    for c, next_c in zip(text, text[1:] + ' '):
        if skip:
            skip -= 1
            result.append(replace_char)
            continue

        pair = (c, next_c)
        if pair == OPEN_COMMENT:
            in_comment += 1
            skip = 1
            c = replace_char
        elif pair == CLOSE_COMMENT:
            in_comment -= 1
            skip = 1
            c = replace_char

        if in_comment > 0:
            result.append(replace_char)
        else:
            result.append(c)

    return ''.join(result)


# TODO improve grammar to make this step not required :(
DEFAULT_PREPROCESSORS = [replace_comments]
_DEFAULT_PREPROCESSORS = object()


def parse_source_code(source_code, *, verbose=0, fn='unknown',
                      preprocessors=_DEFAULT_PREPROCESSORS):
    'Parse source code with the parser'
    if preprocessors is _DEFAULT_PREPROCESSORS:
        preprocessors = DEFAULT_PREPROCESSORS

    processed_source = source_code
    for preprocessor in preprocessors:
        processed_source = preprocessor(processed_source)

    try:
        tree = get_parser().parse(processed_source)
    except Exception as ex:
        if verbose > 1:
            print('[Failure] Parse failure')
            print('-------------------------------')
            print(source_code)
            print('-------------------------------')
            print(f'{type(ex).__name__} {ex}')
            print(f'[Failure] {fn}')
        raise

    if verbose > 2:
        print(f'Successfully parsed {fn}:')
        print('-------------------------------')
        print(source_code)
        print('-------------------------------')
        print(tree.pretty())
        print('-------------------------------')
        print(f'[Success] End of {fn}')

    # This is some WIP comment + declaration matching
    pragmas = RE_PRAGMA.findall(source_code)  # noqa: F841
    line_numbers = _build_map_of_offset_to_line_number(  # noqa: F841
        source_code
    )
    comments = list(RE_COMMENT.finditer(source_code))  # noqa: F841

    # decl, = list(tree.find_data('data_type_declaration'))
    # element1 = [c for c in decl.children][0]
    #
    # print('elem1', element1)
    # # comment 1 at line 9:
    # print(line_numbers[comments[1].end()])
    # # matches up with definition on line 10:
    # print(element1.children[0].children[0].line)
    return tree


def _build_map_of_offset_to_line_number(source):
    '''
    For a multiline source file, return {character_pos: line}
    '''
    start_index = 0
    index_to_line_number = {}
    # A slow and bad algorithm, but only to be used in parsing declarations
    # which are rather small
    for line_number, line in enumerate(source.splitlines(), 1):
        for index in range(start_index, start_index + len(line) + 1):
            index_to_line_number[index] = line_number
        start_index += len(line) + 1
    return index_to_line_number


def parse_single_file(fn, *, verbose=0):
    'Parse a single source code file'
    source_code = get_source_code(fn)
    return parse_source_code(source_code, fn=fn, verbose=verbose)


def parse_project(tsproj_project, *, print_filenames=None, verbose=0):
    'Parse an entire tsproj project file'
    proj_path = pathlib.Path(tsproj_project)
    proj_root = proj_path.parent.resolve().absolute()  # noqa: F841 TODO

    if proj_path.suffix.lower() not in ('.tsproj', ):
        raise ValueError('Expected a .tsproj file')

    project = pytmc.parser.parse(proj_path)
    results = {}
    success = True
    for i, plc in enumerate(project.plcs, 1):
        source_items = (
            list(plc.dut_by_name.items()) +
            list(plc.gvl_by_name.items()) +
            list(plc.pou_by_name.items())
        )
        for name, source_item in source_items:
            if not hasattr(source_item, 'get_source_code'):
                continue

            if print_filenames is not None:
                print(f'* Parsing {source_item.filename}',
                      file=print_filenames)
            source_code = source_item.get_source_code()
            if not source_code:
                continue

            # if '<?xml ' in source_code.splitlines()[0]:
            #     print('found xml')
            #     if name in plc.gvl_by_name or name in plc.dut_by_name:
            #         # TODO pytmc
            #         source_code = source_item.declaration
            #     else:
            #         print('* TODO?', name, source_code)
            #         continue

            try:
                results[name] = parse_source_code(
                    source_code, fn=source_item.filename,
                    verbose=verbose)
            except Exception as ex:
                results[name] = ex
                ex.filename = source_item.filename
                success = False

    return success, results


def build_arg_parser(argparser=None):
    if argparser is None:
        argparser = argparse.ArgumentParser()

    argparser.description = DESCRIPTION
    argparser.formatter_class = argparse.RawTextHelpFormatter

    argparser.add_argument(
        'filename', type=str,
        help=(
            'Path to project, solution, source code file (.tsproj, .sln, '
            '.TcPOU, .TcGVL)'
        )
    )

    argparser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity, up to -vvv'
    )

    argparser.add_argument(
        '--debug',
        action='store_true',
        help='On failure, still return the results tree'
    )

    return argparser


def main(filename, verbose=0, debug=False):
    '''
    Parse the given source code/project.
    '''

    path = pathlib.Path(filename)
    project_fns = []
    source_fns = []
    if path.suffix.lower() in ('.tsproj', ):
        project_fns = [path]
    elif path.suffix.lower() in ('.sln', ):
        project_fns = pytmc.parser.projects_from_solution(path)
    elif path.suffix.lower() in ('.tcpou', '.tcgvl', '.tcdut'):
        source_fns = [path]
    else:
        raise ValueError(f'Expected a tsproj or sln file, got: {path.suffix}')

    results = {}
    success = True
    print_filenames = sys.stdout if verbose > 0 else None

    for fn in project_fns:
        if print_filenames:
            print(f'* Loading project {fn}')
        success, results[fn] = parse_project(
            fn, print_filenames=print_filenames, verbose=verbose)

    for fn in source_fns:
        if print_filenames:
            print(f'* Parsing {fn}')
        try:
            results[fn] = parse_single_file(fn, verbose=verbose)
        except Exception:
            success = False

    def find_failures(res):
        for name, item in res.items():
            if isinstance(item, Exception):
                yield name, item
            elif isinstance(item, dict):
                yield from find_failures(item)

    if not success:
        print('Failed to parse all source code files:')
        failures = list(find_failures(results))
        for name, item in failures:
            fn = f'[{item.filename}] ' if hasattr(item, 'filename') else ''
            header = f'{fn}{name}'
            print(header)
            print('-' * len(header))
            print(f'({type(item).__name__}) {item}')
            print()

        if not debug:
            sys.exit(1)

    return results
