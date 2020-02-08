import re
import sys
import lark

with open('iec.lark', 'rt') as f:
    lark_grammar = f.read()

# iec_parser = lark.Lark(lark_grammar, parser='lalr')
iec_parser = lark.Lark(lark_grammar, parser='earley')

try:
    fn = sys.argv[1]
except IndexError:
    fn = 'types.EXP'

with open(fn) as f:
    source_code = f.read()

tree = iec_parser.parse(source_code)

re_comment = re.compile(r'(//.*$|\(\*.*?\*\))', re.MULTILINE | re.DOTALL)
re_pragma = re.compile(r'{[^}]*?}', re.MULTILINE | re.DOTALL)

pragmas = re_pragma.findall(source_code)


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


line_numbers = _build_map_of_offset_to_line_number(source_code)

comments = list(re_comment.finditer(source_code))

decl, = list(tree.find_data('data_type_declaration'))
element1 = [c for c in decl.children][0]

print('elem1', element1)
# comment 1 at line 9:
print(line_numbers[comments[1].end()])
# matches up with definition on line 10:
print(element1.children[0].children[0].line)
