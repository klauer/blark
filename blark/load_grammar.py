'''
Load the IEC grammar from Volker's original format + convert to Lark grammar

Eventually, this will go away and there will be only lark...
'''

import re
import pathlib

import lark


parser = lark.Lark(
    r'''
    start               : (production | comment)+
    quoted_string       : (SINGLE_QUOTED | DOUBLE_QUOTED)
    keyword             : "$'" /[^']+/ "'"
    SINGLE_QUOTED       : "'" /[^']+/ "'"
    DOUBLE_QUOTED       : "\"" /[^"]+/ "\""

    IDENTIFIER          : /[a-z_][a-z0-9_]*/si
    regex               : "<" /((\\\>)|[^>])+/ ">"

    production          : IDENTIFIER "::=" (one_of | concat) ";"
    concat              : ( regex | keyword | quoted_string | IDENTIFIER | parentheses | zero_or_more | optional | comment )+
    one_of              : concat ( "|" concat )+

    _extended_structure : one_of | regex | keyword | quoted_string | IDENTIFIER | parentheses | zero_or_more | optional | comment

    parentheses         : "(" _extended_structure+ ")"
    zero_or_more        : "{" _extended_structure+ "}"
    optional            : "[" _extended_structure+ "]"

    comment             : /^#\s*(.*)$/m

    // Ignore whitespace
    %import common.WS
    %ignore WS
    ''',
    # debug=True,
)



MODULE_PATH = pathlib.Path(__file__).parent


change_to_terminal = '''
_identifier
type_sint
type_int
type_dint
type_lint
type_usint
type_uint
type_udint
type_ulint
type_real
type_lreal
type_date
type_time
type_bool
type_byte
type_word
type_dword
type_lword
r_edge
f_edge
persistent
string
wstring
read_write
read_only
il_operator_ld
il_operator_ldn
il_operator_st
il_operator_stn
il_operator_not
il_operator_s
il_operator_r
il_operator_s1
il_operator_r1
il_operator_clk
il_operator_cu
il_operator_cd
il_operator_pv
il_operator_in
il_operator_pt
il_operator_or
il_operator_xor
il_operator_orn
il_operator_xorn
il_operator_add
il_operator_sub
il_operator_mul
il_operator_div
il_operator_mod
il_operator_gt
il_operator_ge
il_operator_eq
il_operator_lt
il_operator_le
il_operator_ne
il_operator_cal
il_operator_calc
il_operator_calcn
il_operator_ret
il_operator_retc
il_operator_retcn
il_operator_jmp
il_operator_jmpc
il_operator_jmpcn
logical_or
logical_xor
logical_and
logical_not
modulo
'''.strip().splitlines()


def parenthesize(s):
    if (s[0], s[-1]) == ('"', '"') and s.count('"') == 2:
        return s
    if (s[0], s[-1]) == ('(', ')') and s.count('(') == 1:
        return s
    if (s[0], s[-1]) == ('/', '/') and s.count('/') == 2:
        return s
    if re.match('^[A-Za-z0-9_]+$', s):
        return s
    return f'({s})'


class GrammarTransformer(lark.Transformer):
    def start(self, items):
        return '\n'.join(items).replace('\n\n\n', '\n')

    def concat(self, items):
        if len(items) == 1 and len(items[0]) >= 2:
            return parenthesize(items[0])
        return '(' + ' '.join(str(item) for item in items) + ')'

    def IDENTIFIER(self, identifier):
        if identifier in change_to_terminal:
            if identifier.startswith('_'):
                identifier = identifier[1:]
            return identifier.upper()
        return identifier

    def regex(self, items):
        return f'/{items[0]}/'

    def quoted_string(self, items):
        s = items[0]
        if s.startswith("'"):
            s = s.strip("'")
        elif s.startswith('"'):
            s = s.strip('"')

        flags = 'i' if re.match('[A-Za-z]', s) else ''
        return f'"{s}"' + flags

    def keyword(self, items):
        assert len(items) == 1
        key = items[0].strip("'")
        flags = 'i' if re.match('[A-Za-z]', key) else ''
        return f'"{key}"' + flags  # i makes it case insensitive

    def comment(self, items):
        # return ''
        comment = items[0].lstrip(' #\n\r')
        return f'\n// {comment}'

    def optional(self, items):
        items = ' '.join(items)
        return parenthesize(items) + '?'
        # return f'({items})?'

    def zero_or_more(self, items):
        items = ' '.join(items)
        return parenthesize(items) + '*'
        # return f'({items})*'

    def parentheses(self, items):
        items = ' '.join(items)
        return parenthesize(items)
        # return f'({items})'

    def one_of(self, items):
        return '(' + ' | '.join(str(item) for item in items) + ')'

    def production(self, items):
        assert len(items) == 2
        name, value = items
        if value[0] == '(' and value[-1] == ')':
            value = value[1:-1]
        return '\n' + name + ': ' + value


def convert_to_lark(orig_grammar):
    'Original grammar -> Lark grammar'
    tree = parser.parse(orig_grammar)


    transformer = GrammarTransformer()
    lark_grammar = '// (Auto-generated Lark grammar from iec.grammar)\n'
    lark_grammar += transformer.transform(tree)
    lark_grammar += r'''

    start: iec_source

    MULTI_LINE_COMMENT: /\(\*.*?\*\)/s
    SINGLE_LINE_COMMENT: /\s*/ "//" /[^\n]/*

    // Ignore whitespace
    %import common.WS
    %ignore WS
    %ignore MULTI_LINE_COMMENT
    %ignore SINGLE_LINE_COMMENT
    '''

# (TODO)
# EOL:
# CR : /\r/
# LF : /\n/
# NEWLINE: (CR? LF)+
    return lark_grammar


def main():
    orig_path = MODULE_PATH / 'iec.grammar'
    lark_path = MODULE_PATH / 'iec.lark'
    print(f'Loading original grammar from: {orig_path}...')
    with open(orig_path) as f:
        orig_grammar = f.read()

    print(f'Converting to lark grammar...')
    lark_grammar = convert_to_lark(orig_grammar)

    print(f'Saving lark grammar to: {lark_path}...')

    with open(lark_path, 'wt') as f:
        print(lark_grammar, file=f)

    return lark_grammar


if __name__ == '__main__':
    main()
