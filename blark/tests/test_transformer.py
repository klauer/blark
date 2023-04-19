import pathlib
from typing import List, Optional

import pytest
from pytest import param

from .. import transform as tf
from ..parse import parse_source_code
from . import conftest

TEST_PATH = pathlib.Path(__file__).parent


def roundtrip_rule(rule_name: str, value: str, expected: Optional[str] = None):
    """
    Round-trip a blark grammar rule for testing purposes.

    1. Parse and transform the source code in ``value`` into dataclasses
    2. Ensure that the dataclass source code representation is identical
       to the input
    3. Run serialization/deserialization checks (if enabled)
    """
    parser = conftest.get_grammar(start=rule_name)
    transformed = parse_source_code(value, parser=parser)
    print("\n\nTransformed:")
    print(repr(transformed))
    print("\n\nOr:")
    print(transformed)
    if expected is None:
        expected = value
    try:
        assert str(transformed) == expected, \
            "Transformed object does not produce identical source code"
    except Exception:
        tree = parse_source_code(value, parser=parser, transform=False)
        print("\n\nTransformation failure. The original source code was:")
        print(value)
        print("\n\nThe parse tree is:")
        print(tree.pretty())
        raise

    conftest.check_serialization(
        transformed, deserialize=True, require_same_source=True
    )
    return transformed


def test_check_unhandled_rules(grammar):
    defined_rules = set(
        rule.origin.name for rule in grammar.rules
        if not rule.origin.name.startswith("_")
        and not rule.options.expand1
    )
    transformer = tf.GrammarTransformer()
    unhandled_rules = set(
        str(name)
        for name in defined_rules
        if not hasattr(transformer, name)
    )

    handled_separately = {
        # no individual ones for time
        "days",
        "hours",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
        "minutes",

        # handled as aliases
        "case_list",

        # handled as special cases
        "array_initialization",

        # handled as tree
        "global_var_list",
        "var_body",
    }

    todo_rules = set()

    aliased = {
        "boolean_literal",
        "fb_decl",
    }

    assert set(unhandled_rules) == handled_separately | todo_rules | aliased


@pytest.mark.parametrize(
    "name, value, expected",
    [
        param("integer_literal", "-12", tf.Integer(value="-12")),
        param("integer_literal", "12", tf.Integer(value="12")),
        param("integer_literal", "10#12", tf.Integer(value="12")),
        param("integer_literal", "INT#12", tf.Integer(value="12", type_name="INT")),
        param("integer_literal", "2#10010", tf.BinaryInteger(value="10010")),
        param("integer_literal", "8#22", tf.OctalInteger(value="22")),
        param("integer_literal", "16#12", tf.HexInteger(value="12")),
        param("integer_literal", "UDINT#12", tf.Integer(value="12", type_name="UDINT")),
        param("integer_literal", "UDINT#2#010", tf.BinaryInteger(value="010", type_name="UDINT")),  # noqa: E501
        param("integer_literal", "UDINT#2#1001_0011", tf.BinaryInteger(value="1001_0011", type_name="UDINT")),  # noqa: E501
        param("integer_literal", "DINT#16#C0FFEE", tf.HexInteger(value="C0FFEE", type_name="DINT")),  # noqa: E501
        param("real_literal", "-12.0", tf.Real(value="-12.0")),
        param("real_literal", "12.0", tf.Real(value="12.0")),
        param("real_literal", "12.0e5", tf.Real(value="12.0e5")),
        param("bit_string_literal", "1234", tf.BitString(type_name=None, value="1234")),
        param("bit_string_literal", "WORD#1234", tf.BitString(type_name="WORD", value="1234")),
        param("bit_string_literal", "WORD#2#0101", tf.BinaryBitString(type_name="WORD", value="0101")),  # noqa: E501
        param("bit_string_literal", "WORD#8#777", tf.OctalBitString(type_name="WORD", value="777")),  # noqa: E501
        param("bit_string_literal", "word#16#FEEE", tf.HexBitString(type_name="word", value="FEEE")),  # noqa: E501
        param("duration", "TIME#-1D", tf.Duration(days="1", negative=True)),
        param("duration", "TIME#1D", tf.Duration(days="1")),
        param("duration", "TIME#10S", tf.Duration(seconds="10")),
        param("duration", "TIME#1H", tf.Duration(hours="1")),
        param("duration", "TIME#1M", tf.Duration(minutes="1")),
        param("duration", "TIME#10MS", tf.Duration(milliseconds="10")),
        param("duration", "TIME#1.1D", tf.Duration(days="1.1")),
        param("duration", "TIME#10.1S", tf.Duration(seconds="10.1")),
        param("duration", "TIME#1.1H", tf.Duration(hours="1.1")),
        param("duration", "TIME#1.1M", tf.Duration(minutes="1.1")),
        param("duration", "TIME#10.1MS", tf.Duration(milliseconds="10.1")),
        param("duration", "T#1D1H1M1S1MS", tf.Duration(days="1", hours="1", minutes="1", seconds="1", milliseconds="1")),  # noqa: E501
        param("duration", "TIME#1H1M1S1MS", tf.Duration(hours="1", minutes="1", seconds="1", milliseconds="1")),  # noqa: E501
        param("time_of_day", "TIME_OF_DAY#1:1:1.2", tf.TimeOfDay(hour="1", minute="1", second="1.2")),  # noqa: E501
        param("lduration", "LTIME#-1D", tf.Lduration(days="1", negative=True)),
        param("lduration", "LTIME#1D", tf.Lduration(days="1")),
        param("lduration", "LTIME#10S", tf.Lduration(seconds="10")),
        param("lduration", "LTIME#1H", tf.Lduration(hours="1")),
        param("lduration", "LTIME#1M", tf.Lduration(minutes="1")),
        param("lduration", "LTIME#10MS", tf.Lduration(milliseconds="10")),
        param("lduration", "LTIME#1.1D", tf.Lduration(days="1.1")),
        param("lduration", "LTIME#10.1S", tf.Lduration(seconds="10.1")),
        param("lduration", "LTIME#1.1H", tf.Lduration(hours="1.1")),
        param("lduration", "LTIME#1.1M", tf.Lduration(minutes="1.1")),
        param("lduration", "LTIME#10.1MS", tf.Lduration(milliseconds="10.1")),
        param("lduration", "LT#1D1H1M1S1MS", tf.Lduration(days="1", hours="1", minutes="1", seconds="1", milliseconds="1")),  # noqa: E501
        param("lduration", "LTIME#1H1M1S1MS", tf.Lduration(hours="1", minutes="1", seconds="1", milliseconds="1")),  # noqa: E501
        param("ltime_of_day", "LTIME_OF_DAY#1:1:1.2", tf.LtimeOfDay(hour="1", minute="1", second="1.2")),  # noqa: E501
    ],
)
def test_literal(name, value, expected):
    transformed = roundtrip_rule(name, value, expected=str(expected))
    assert transformed == expected


@pytest.mark.parametrize(
    "name, value",
    [
        param("integer_literal", "-12"),
        param("integer_literal", "12"),
        param("integer_literal", "INT#12"),
        param("integer_literal", "2#10010"),
        param("integer_literal", "8#22"),
        param("integer_literal", "16#12"),
        param("integer_literal", "UDINT#12"),
        param("integer_literal", "UDINT#2#010"),
        param("integer_literal", "UDINT#2#1001_0011"),
        param("integer_literal", "DINT#16#C0FFEE"),
        param("real_literal", "-12.0"),
        param("real_literal", "12.0"),
        param("real_literal", "12.0e5"),
        param("bit_string_literal", "WORD#1234"),
        param("bit_string_literal", "WORD#2#0101"),
        param("bit_string_literal", "WORD#8#777"),
        param("bit_string_literal", "word#16#FEEE"),
        param("duration", "TIME#1D"),
        param("duration", "TIME#10S"),
        param("duration", "TIME#1H"),
        param("duration", "TIME#1M"),
        param("duration", "TIME#10MS"),
        param("duration", "TIME#1.1D"),
        param("duration", "TIME#10.1S"),
        param("duration", "TIME#1.1H"),
        param("duration", "TIME#1.1M"),
        param("duration", "TIME#10.1MS"),
        param("duration", "TIME#1D1H1M1S1MS"),
        param("duration", "TIME#1H1M1S1MS"),
        param("time_of_day", "TIME_OF_DAY#1:1:1.2"),
        param("date", "DATE#1970-1-1"),
        param("date_and_time", "DT#1970-1-1-1:2:30.3"),
        param("date_and_time", "DT#1970-1-1-0:0:0"),
        param("date_and_time", "DT#1970-1-1-0:0"),
        param("ldate", "LDATE#1970-1-1"),
        param("ldate_and_time", "LDT#1970-1-1-1:2:30.300123456"),
        param("single_byte_string_spec", "STRING[1]"),
        param("single_byte_string_spec", "STRING(1)"),
        param("single_byte_string_spec", "STRING(1) := 'abc'"),
        param("double_byte_string_spec", "WSTRING[1]"),
        param("double_byte_string_spec", "WSTRING(1)"),
        param("double_byte_string_spec", 'WSTRING(1) := "abc"'),
    ],
)
def test_literal_roundtrip(name, value):
    roundtrip_rule(name, value)


@pytest.mark.parametrize(
    "name, value, expected",
    [
        param("boolean_literal", "BOOL#1", "TRUE"),
        param("boolean_literal", "BOOL#0", "FALSE"),
        param("boolean_literal", "BOOL#TRUE", "TRUE"),
        param("boolean_literal", "BOOL#FALSE", "FALSE"),
        param("boolean_literal", "1", "TRUE"),
        param("boolean_literal", "0", "FALSE"),
        param("boolean_literal", "TRUE", "TRUE"),
        param("boolean_literal", "FALSE", "FALSE"),
    ],
)
def test_bool_literal_roundtrip(name, value, expected):
    roundtrip_rule(name, value, expected=expected)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("direct_variable", "%IX1.2"),
        param("direct_variable", "%IX1"),
        param("direct_variable", "%QX1.2"),
        param("direct_variable", "%QX1"),
        param("direct_variable", "%MX1.2"),
        param("direct_variable", "%MX1"),
        param("direct_variable", "%IB1.2"),
        param("direct_variable", "%IW1"),
        param("direct_variable", "%QD1.2"),
        param("direct_variable", "%QL1"),
        param("direct_variable", "%MW1.2"),
        param("direct_variable", "%ML1"),
        param("multi_element_variable", "a.b"),
        param("multi_element_variable", "a^.b"),
        param("multi_element_variable", "a^.b[1, 2]"),
        param("multi_element_variable", "a[1, 2, 3, 4]^.b[1, 2]"),
        param("multi_element_variable", "Abc123[1]^._defGhi"),
        param("expression", "-1"),
        param("expression", "+1"),
        param("expression", "1"),
        param("expression", "1 + 1"),
        param("expression", "1 / 2"),
        param("expression", "3 * 4"),
        param("expression", "3 MOD 4"),
        param("expression", "3 XOR 4"),
        param("expression", "3 = 4"),
        param("expression", "3 <> 4"),
        param("expression", "3 <= 4"),
        param("expression", "3 >= 4"),
        param("expression", "3 ** 4"),
        param("expression", "1 + 2 * (3 - 4)"),
        param("expression", "NOT 1"),
        param("expression", "NOT (3 - 4)"),
        param("expression", "(i_xTrigger OR NOT i_xPress_OK) AND NOT xVeto"),
        param("expression", "(nEventIdx := (nEventIdx + 1)) = nMaxEvents"),
        param("simple_type_declaration", "TypeName : INT"),
        param("simple_type_declaration", "TypeName : INT := 5"),
        param("simple_type_declaration", "TypeName : INT := 5 + 1 * (2)"),
        param("simple_type_declaration", "TypeName : REFERENCE TO INT"),
        param("simple_type_declaration", "TypeName : POINTER TO INT"),
        param("simple_type_declaration", "TypeName : POINTER TO POINTER TO INT"),
        param("simple_type_declaration", "TypeName : REFERENCE TO POINTER TO INT"),
        param("simple_type_declaration", "TypeName EXTENDS a.b : POINTER TO INT"),
        param("subrange_specification", "TypeName"),  # aliased and not usually hit
        param("subrange_type_declaration", "TypeName : INT (1..2)"),
        param("subrange_type_declaration", "TypeName : INT (*) := 1"),
        param("enumerated_type_declaration", "TypeName : TypeName := Value"),
        param("enumerated_type_declaration", "TypeName : (Value1 := 1, Value2 := 2)"),
        param("enumerated_type_declaration", "TypeName : (Value1 := 1, Value2 := 2) INT := Value1"),  # noqa: E501
        param("enumerated_type_declaration", "TypeName : (Value1 := 1, Value2 := 2) INT := Value1"),  # noqa: E501
        param("array_type_declaration", "TypeName : ARRAY [1..2, 3..4] OF INT"),
        param("array_type_declaration", "TypeName : ARRAY [1..2] OF INT := [1, 2]"),
        param("array_type_declaration", "TypeName : ARRAY [1..2, 3..4] OF INT := [2(3), 3(4)]"),
        param("array_type_declaration", "TypeName : ARRAY [1..2, 3..4] OF Tc.SomeType"),
        param("array_type_declaration", "TypeName : ARRAY [1..2, 3..4] OF Tc.SomeType(someInput := 3)"),  # noqa: E501
        param("structure_type_declaration", "TypeName :\nSTRUCT\nEND_STRUCT"),
        param("structure_type_declaration", "TypeName EXTENDS Other.Type :\nSTRUCT\nEND_STRUCT"),
        param("structure_type_declaration", "TypeName : POINTER TO\nSTRUCT\nEND_STRUCT"),
        param("structure_type_declaration", tf.multiline_code_block(
            """
            TypeName : POINTER TO
            STRUCT
                iValue : INT;
            END_STRUCT
            """
        )),
        param("structure_type_declaration", tf.multiline_code_block(
            """
            TypeName : POINTER TO
            STRUCT
                iValue : INT := 3 + 4;
                stTest : ST_Testing := (1, 2);
                eValue : E_Test := E_Test.ABC;
                arrValue : ARRAY [1..2] OF INT := [1, 2];
                arrValue1 : INT (1..2);
                arrValue1 : (Value1 := 1) INT;
                sValue : STRING := 'abc';
                iValue1 AT %I* : INT := 5;
                iValue2 AT %Q* : INT := 5;
                iValue3 AT %M* : INT := 5;
                sValue1 : STRING[10] := 'test';
            END_STRUCT
            """
        )),
        param("string_type_declaration", "TypeName : STRING"),
        param("string_type_declaration", "TypeName : STRING := 'literal'"),
        param("string_type_declaration", "TypeName : STRING[5]"),
        param("string_type_declaration", "TypeName : STRING[100] := 'literal'"),
        param("string_type_declaration", 'TypeName : WSTRING[100] := "literal"'),
    ],
)
def test_expression_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [

        param("location", "AT %IX1.1"),
        param("var1", "iValue AT %IX1.1"),
        param("var1", "iValue AT %I*"),
        param("var1", "iValue AT %Q*"),
        param("var1", "iValue AT %M*"),
        param("var1", "iValue"),
        param("edge_declaration", "iValue AT %IX1.1 : BOOL R_EDGE"),
        param("edge_declaration", "iValue : BOOL F_EDGE"),
        # param("array_var_init_decl", ""),
        param("input_declarations", tf.multiline_code_block(
            """
            VAR_INPUT
            END_VAR
            """
        )),
        param("input_declarations", tf.multiline_code_block(
            """
            VAR_INPUT RETAIN
            END_VAR
            """
        )),
        param("input_declarations", tf.multiline_code_block(
            """
            VAR_INPUT RETAIN
                iValue : INT;
                sValue : STRING := 'abc';
                wsValue : WSTRING := "abc";
            END_VAR
            """
        )),
        param("input_declarations", tf.multiline_code_block(
            """
            VAR_INPUT RETAIN
                iValue : INT;
                sValue : STRING := 'abc';
                wsValue : WSTRING := "abc";
                fbTest : FB_Test(1, 2, 3);
                fbTest : FB_Test(A := 1, B := 2, C => 3);
                fbTest : FB_Test(1, 2, A := 1, B := 2, C => 3);
                fbTest : FB_Test := (1, 2, 3);
            END_VAR
            """
        )),
        param("input_declarations", tf.multiline_code_block(
            """
            VAR_INPUT RETAIN
                fbTest : FB_Test := (A := 1, B := 2, C := 3);
            END_VAR
            """,
            ),
        ),
    ],
)
def test_input_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("output_declarations", tf.multiline_code_block(
            """
            VAR_OUTPUT
            END_VAR
            """
        )),
        param("output_declarations", tf.multiline_code_block(
            """
            VAR_OUTPUT RETAIN
            END_VAR
            """
        )),
        param("output_declarations", tf.multiline_code_block(
            """
            VAR_OUTPUT RETAIN
                iValue : INT;
                sValue : STRING := 'abc';
                wsValue : WSTRING := "abc";
            END_VAR
            """
        )),
        param("output_declarations", tf.multiline_code_block(
            """
            VAR_OUTPUT RETAIN
                iValue : INT;
                sValue : STRING := 'abc';
                wsValue : WSTRING := "abc";
                fbTest : FB_Test(1, 2, 3);
                fbTest : FB_Test(A := 1, B := 2, C => 3);
                fbTest : FB_Test(1, 2, A := 1, B := 2, C => 3);
                fbTest : FB_Test := (1, 2, 3);
            END_VAR
            """
        )),
        param("output_declarations", tf.multiline_code_block(
            """
            VAR_OUTPUT RETAIN
                fbTest : FB_Test := (A := 1, B := 2, C := 3);
            END_VAR
            """,
            ),
        ),
    ],
)
def test_output_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("input_output_declarations", tf.multiline_code_block(
            """
            VAR_IN_OUT
            END_VAR
            """
        )),
        param("input_output_declarations", tf.multiline_code_block(
            """
            VAR_IN_OUT
                iValue : INT;
                sValue : STRING := 'abc';
                wsValue : WSTRING := "abc";
            END_VAR
            """
        )),
        param("input_output_declarations", tf.multiline_code_block(
            """
            VAR_IN_OUT
                iValue : INT;
                sValue : STRING := 'abc';
                wsValue : WSTRING := "abc";
                fbTest : FB_Test(1, 2, 3);
                fbTest : FB_Test(A := 1, B := 2, C => 3);
                fbTest : FB_Test(1, 2, A := 1, B := 2, C => 3);
                fbTest : FB_Test(initializer := 5) := (A := 1, B := 2, C := 3);
                fbTest : FB_Test := (1, 2, 3);
            END_VAR
            """
        )),
        param("input_output_declarations", tf.multiline_code_block(
            """
            VAR_IN_OUT
                fbProblematic1 : FB_Test(initializer := 5) := (A := 1, B := 2, C := 3);
            END_VAR
            """
        )),
        param("input_output_declarations", tf.multiline_code_block(
            """
            VAR_IN_OUT
                fbTest : FB_Test := (A := 1, B := 2, C := 3);
            END_VAR
            """,
            ),
        ),
    ],
)
def test_input_output_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("program_access_decls", tf.multiline_code_block(
            """
            VAR_ACCESS
                AccessName : SymbolicVariable : TypeName READ_WRITE;
                AccessName1 : SymbolicVariable1 : TypeName1 READ_ONLY;
                AccessName2 : SymbolicVariable2 : TypeName2;
            END_VAR
            """
        )),
    ],
)
def test_var_access_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param(
            "static_var_declarations",
            tf.multiline_code_block(
                """
                VAR_STAT
                    iValue : INT := 1;
                END_VAR
                """
            ),
        ),
        param(
            "static_var_declarations",
            tf.multiline_code_block(
                """
                VAR_STAT CONSTANT
                    iValue : INT := 1;
                END_VAR
                """
            ),
        ),
    ],
)
def test_var_stat_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL
            END_VAR
            """
        )),
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL INTERNAL
            END_VAR
            """
        )),
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL CONSTANT
            END_VAR
            """
        )),
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL PERSISTENT
            END_VAR
            """
        )),
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL CONSTANT PERSISTENT
            END_VAR
            """
        )),
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL CONSTANT PERSISTENT
                iValue : INT := 5;
                fbTest1 : FB_Test(1, 2);
                fbTest2 : FB_Test(A := 1, B := 2);
                fbTest3 : FB_TestC;
                i_iFscEB1Ch0AI AT %I* : INT;
            END_VAR
            """
        )),
    ],
)
def test_global_roundtrip(rule_name: str, value: str):
    gvl = roundtrip_rule(rule_name, value)

    print(gvl.attribute_pragmas)


@pytest.mark.parametrize(
    "rule_name, value, pragmas",
    [
        param(
            "global_var_declarations",
            tf.multiline_code_block(
                """
                VAR_GLOBAL CONSTANT PERSISTENT
                    iValue : INT := 5;
                END_VAR
                """,
            ),
            [],
            id="no_attrs",
        ),
        param(
            "global_var_declarations",
            tf.multiline_code_block(
                """
                {attribute abc}
                VAR_GLOBAL CONSTANT PERSISTENT
                    iValue : INT := 5;
                END_VAR
                """,
            ),
            ["abc"],
            id="one_attr",
        ),
        param(
            "global_var_declarations",
            tf.multiline_code_block(
                """
                // Line one
                {attribute abc}
                // Line two
                {attribute def}
                VAR_GLOBAL CONSTANT PERSISTENT
                    iValue : INT := 5;
                END_VAR
                """,
            ),
            ["abc", "def"],
            id="two_attrs",
        ),
    ],
)
def test_global_attr_pragmas(rule_name: str, value: str, pragmas: List[str]):
    gvl = roundtrip_rule(rule_name, value)
    assert gvl.attribute_pragmas == pragmas


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("non_generic_type_name", "POINTER TO FBName"),
        param("non_generic_type_name", "POINTER TO POINTER TO FBName"),
        param("non_generic_type_name", "FBName"),
        param("non_generic_type_name", "POINTER TO Package.FBName"),
        param("non_generic_type_name", "POINTER TO POINTER TO Package.FBName"),
        param("non_generic_type_name", "Package.FBName"),
    ],
)
def test_type_name_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName IMPLEMENTS I_fbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName IMPLEMENTS I_fbName, I_fbName2
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK ABSTRACT fbName EXTENDS OtherFbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK PRIVATE fbName EXTENDS OtherFbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK PUBLIC fbName EXTENDS OtherFbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK INTERNAL fbName EXTENDS OtherFbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK PROTECTED fbName EXTENDS OtherFbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK FINAL fbName EXTENDS OtherFbName
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName
            VAR_INPUT
                bExecute : BOOL;
            END_VAR
            VAR_OUTPUT
                iResult : INT;
            END_VAR
            VAR_IN_OUT
                iShared : INT;
            END_VAR
            VAR CONSTANT
                iConstant : INT := 5;
            END_VAR
            VAR
                iInternal : INT;
            END_VAR
            VAR RETAIN
                iRetained : INT;
            END_VAR
            END_FUNCTION_BLOCK
            """
        )),
        param("located_var_declarations", tf.multiline_code_block(
            """
            VAR RETAIN
                iValue AT %IB1 : INT := 5;
            END_VAR
            """
        )),
        param("external_var_declarations", tf.multiline_code_block(
            """
            VAR_EXTERNAL
                iGlobalVar : INT;
            END_VAR
            """
        )),
        param("external_var_declarations", tf.multiline_code_block(
            """
            VAR_EXTERNAL CONSTANT
                iGlobalVar : INT;
            END_VAR
            """
        )),
        param("temp_var_decls", tf.multiline_code_block(
            """
            VAR_TEMP
                iGlobalVar : INT;
            END_VAR
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName
                iValue := 1;
                iValue;
                iValue S= 1;
                iValue R= 1;
                iValue REF= GVL.iTest;
                fbOther(A := 5, B => iValue, NOT C => iValue1);
                IF 1 THEN
                    iValue := 1;
                    IF 1 THEN
                        iValue := 1;
                    END_IF
                END_IF
                Method();
                RETURN;
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName
                Method();
                IF 1 THEN
                    EXIT;
                END_IF
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_type_declaration", tf.multiline_code_block(
            """
            FUNCTION_BLOCK fbName
                Method();
                IF 1 THEN
                    CONTINUE;
                END_IF
            END_FUNCTION_BLOCK
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD PRIVATE MethodName : RETURNTYPE
            END_METHOD
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD PRIVATE MethodName : ARRAY [1..2] OF INT
            END_METHOD
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD PUBLIC MethodName : RETURNTYPE
            END_METHOD
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD PUBLIC MethodName
            END_METHOD
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD MethodName : RETURNTYPE
                VAR_INPUT
                    bExecute : BOOL;
                END_VAR
                VAR_OUTPUT
                    iResult : INT;
                END_VAR
                iResult := 5;
            END_METHOD
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD PUBLIC MethodName
                VAR_INST
                    bExecute : BOOL;
                END_VAR
            END_METHOD
            """
        )),
        param("function_block_method_declaration", tf.multiline_code_block(
            """
            METHOD PUBLIC ABSTRACT MethodName : LREAL
                VAR_INPUT
                    I : UINT;
                END_VAR
            END_METHOD
            """
        )),
        param("function_block_property_declaration", tf.multiline_code_block(
            """
            PROPERTY PRIVATE PropertyName : RETURNTYPE
            END_PROPERTY
            """
        )),
        param("function_block_property_declaration", tf.multiline_code_block(
            """
            PROPERTY PRIVATE PropertyName : ARRAY [1..2] OF INT
            END_PROPERTY
            """
        )),
        param("function_block_property_declaration", tf.multiline_code_block(
            """
            PROPERTY PUBLIC PropertyName : RETURNTYPE
            END_PROPERTY
            """
        )),
        param("function_block_property_declaration", tf.multiline_code_block(
            """
            PROPERTY PUBLIC PropertyName
            END_PROPERTY
            """
        )),
        param("function_block_property_declaration", tf.multiline_code_block(
            """
            PROPERTY PropertyName : RETURNTYPE
                VAR_INPUT
                    bExecute : BOOL;
                END_VAR
                VAR_OUTPUT
                    iResult : INT;
                END_VAR
                iResult := 5;
            END_PROPERTY
            """
        )),
        param("function_block_property_declaration", tf.multiline_code_block(
            """
            PROPERTY PUBLIC ABSTRACT PropertyName : LREAL
                VAR_INPUT
                    I : UINT;
                END_VAR
            END_PROPERTY
            """
        )),
        param("fb_decl", tf.multiline_code_block(
            """
            fb1, fb2 : TypeName
            """
        )),
        param("fb_decl", tf.multiline_code_block(
            """
            fb1, fb2 : TypeName := (1, 2)
            """
        )),
    ],
)
def test_fb_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param(
            "action",
            tf.multiline_code_block(
                """
                ACTION actActionName:
                END_ACTION
                """
            ),
            id="empty_action",
        ),
        param(
            "action",
            tf.multiline_code_block(
                """
                ACTION actActionName:
                    iValue := iValue + 1;
                END_ACTION
                """
            ),
            id="simple_action",
        ),
    ],
)
def test_action_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("if_statement", tf.multiline_code_block(
            """
            IF 1 THEN
                iValue := 1;
                IF 1 THEN
                    iValue := 1;
                END_IF
            ELSIF 2 THEN
                iValue := 2;
            ELSIF 3 THEN
                iValue := 2;
            ELSE
                iValue := 3;
            END_IF
            """
        )),
        param("if_statement", tf.multiline_code_block(
            """
            IF 1 THEN
                IF 2 THEN
                    IF 3 * x THEN
                        y();
                    ELSE
                    END_IF
                END_IF
            END_IF
            """
        )),
        param("if_statement", tf.multiline_code_block(
            """
            IF callable() THEN
                y();
            END_IF
            """
        )),
        param("if_statement", tf.multiline_code_block(
            """
            IF a() AND b(in := 5) <> 0 AND c(in1 := expr, in2 := 'test') THEN
                y();
            END_IF
            """
        )),
        param("if_statement", tf.multiline_code_block(
            """
            IF 1 AND_THEN 1 THEN
                y();
            END_IF
            """
        )),
        param("if_statement", tf.multiline_code_block(
            """
            IF 0 OR_ELSE 1 THEN
                y();
            END_IF
            """
        )),
        param("case_statement", tf.multiline_code_block(
            """
            CASE expr OF
            1:
                abc();
            2, 3, GVL.Constant:
                def();
            ELSE
                ghi();
            END_CASE
            """
        )),
        param("case_statement", tf.multiline_code_block(
            """
            CASE a.b.c^.d OF
            1..10:
                OneToTen := OneToTen + 1;
            EnumValue:
            END_CASE
            """
        )),
        param("case_statement", tf.multiline_code_block(
            """
            CASE expr OF
            BYTE#9..BYTE#10, BYTE#13, BYTE#28:
                OneToTen := OneToTen + 1;
            EnumValue:
            END_CASE
            """
        )),
        param("case_statement", tf.multiline_code_block(
            """
            CASE expr OF
            TRUE:
                abc();
            FALSE:
                def();
            END_CASE
            """
        )),
        param("while_statement", tf.multiline_code_block(
            """
            WHILE expr
            DO
                iValue := iValue + 1;
            END_WHILE
            """
        )),
        param("repeat_statement", tf.multiline_code_block(
            """
            REPEAT
                iValue := iValue + 1;
            UNTIL expr
            END_REPEAT
            """
        )),
        param("for_statement", tf.multiline_code_block(
            """
            FOR iIndex := 0 TO 10
            DO
                iValue := iIndex * 2;
            END_FOR
            """
        )),
        param("for_statement", tf.multiline_code_block(
            """
            FOR iIndex := 0 TO 10 BY 1
            DO
                iValue := iIndex * 2;
            END_FOR
            """
        )),
        param("for_statement", tf.multiline_code_block(
            """
            FOR iIndex := (iValue - 5) TO iValue * 10 BY iValue MOD 10
            DO
                arrArray[iIndex] := iIndex * 2;
            END_FOR
            """
        )),
        param("for_statement", tf.multiline_code_block(
            """
            FOR iIndex[1] := 0 TO 10
            DO
                iValue := iIndex * 2;
            END_FOR
            """
        )),
        param("chained_function_call_statement", tf.multiline_code_block(
            """
            uut.call1().call2().call3().call4().done();
            """
        )),
        param("chained_function_call_statement", tf.multiline_code_block(
            """
            uut.call1()^.call2().call3()^.call4().done();
            """
        )),
        param("chained_function_call_statement", tf.multiline_code_block(
            """
            uut.call1()^.call2(A := 1).call3(B := 2)^.call4().done();
            """
        )),
    ],
)
def test_statement_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION FuncName : INT
                VAR_INPUT
                    iValue : INT := 0;
                END_VAR
                FuncName := iValue;
            END_FUNCTION
            """),
            id="int_with_input",
        ),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION INTERNAL FuncName : INT
                VAR_INPUT
                    iValue : INT := 0;
                END_VAR
                FuncName := iValue;
            END_FUNCTION
            """),
            id="int_with_input",
        ),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION FuncName : POINTER TO INT
                VAR
                    iValue : INT := 0;
                END_VAR
                FuncName := ADR(iValue);
            END_FUNCTION
            """),
            id="int_with_pointer_retval",
        ),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION FuncName : INT
                VAR_INPUT
                    iValue : INT := 0;
                END_VAR
                VAR_OUTPUT
                    iOutput : INT;
                END_VAR
                VAR
                    iVar : INT;
                END_VAR
                VAR CONSTANT
                    iVarConst : INT := 123;
                END_VAR
                FuncName := iValue;
            END_FUNCTION
            """),
            id="int_with_input_output",
          ),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION FuncName
                VAR_INPUT
                    Ptr : POINTER TO UINT;
                END_VAR
                Ptr^ := 5;
            END_FUNCTION
            """),
            id="no_return_type",
        ),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION FuncName : BOOL
                VAR
                    _c_epoch : dot.dateTime_t := (dateTime := DT#1970-1-1-0:0:0, uSec := 0);
                END_VAR
            END_FUNCTION
            """),
            id="default_with_datetime",
        ),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION FuncName : Tc2_System.T_MaxString
                VAR_INPUT
                    Ptr : POINTER TO UINT;
                END_VAR
                Ptr^ := 5;
            END_FUNCTION
            """),
            id="dotted_return_type",
        ),
    ],
)
def test_function_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("program_declaration", tf.multiline_code_block(
            """
            PROGRAM ProgramName
            END_PROGRAM
            """
        )),
        param("program_declaration", tf.multiline_code_block(
            """
            PROGRAM ProgramName
                VAR_INPUT
                    iValue : INT;
                END_VAR
                VAR_ACCESS
                    AccessName : SymbolicVariable : TypeName READ_WRITE;
                END_VAR
                iValue := iValue + 1;
            END_PROGRAM
            """
        )),
    ],
)
def test_program_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("input_output_declarations", tf.multiline_code_block(
            """
            // Var in and out
            (* Var in and out *)
            VAR_IN_OUT
                // Variable
                iVar : INT;
            END_VAR
            """
        )),
    ],
)
def test_input_output_comments(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("incomplete_located_var_declarations", tf.multiline_code_block(
            """
            VAR
                iValue AT %Q* : INT;
                sValue AT %I* : STRING [255];
                wsValue AT %I* : WSTRING [255];
            END_VAR
            """
        )),
        param("incomplete_located_var_declarations", tf.multiline_code_block(
            """
            VAR RETAIN
                iValue AT %I* : INT;
                iValue1 AT %Q* : INT;
            END_VAR
            """
        )),
    ],
)
def test_incomplete_located_var_decls(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE TypeName :
                STRUCT
                    xPLC_CnBitsValid : BOOL;
                    xPLC_CnBits : ARRAY [0..20] OF BYTE;
                END_STRUCT
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE ArrayTypeName : ARRAY [1..10, 2..20] OF INT;
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE StringTypeName : STRING[10] := 'Literal';
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE SimpleTypeName EXTENDS OtherType : POINTER TO INT;
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE SubrangeTypeName : POINTER TO INT (1..5);
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE EnumeratedTypeName : REFERENCE TO Identifier;
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE EnumeratedTypeName : REFERENCE TO (IdentifierA, INT#IdentifierB := 1);
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE TypeName :
                UNION
                    intval : INT;
                    as_bytes : ARRAY [0..2] OF BYTE;
                END_UNION
            END_TYPE
            """
        )),
        param(
            "data_type_declaration",
            tf.multiline_code_block(
                """
            TYPE TypeName :
                UNION
                END_UNION
            END_TYPE
            """
            ),
            id="empty_union",   # TODO: this may not be grammatical
        ),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE TypeName :
                UNION
                    intval : INT;
                    enum : (iValue := 1, iValue2 := 2) INT;
                END_UNION
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE TypeName :
                UNION
                    pt_value : POINTER TO INT;
                    pt_SomethingElse : POINTER TO class_SomethingCool;
                END_UNION
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE TypeName :
                STRUCT
                    object : class_InitializeWithFBInit(input := one);
                    interval : INT;
                END_STRUCT
            END_TYPE
            """
        )),
        param("data_type_declaration", tf.multiline_code_block(
            """
            TYPE INTERNAL EnumTypeName : (ENUM_VAL_1 := 0, ENUM_VAL_2);
            END_TYPE
            """
        )),
    ],
)
def test_data_type_declaration(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("structure_element_initialization", "1"),
        param("structure_element_initialization", "name := 1"),
        param("structure_element_initialization", "name := 1 + 2"),
        param("structure_element_initialization", "name := GVL.Constant"),
        param("structure_element_initialization", "name := [1, 2, 3]"),
        # hmm - aliased by array_initialization?
        # param("structure_element_initialization", "name := (a:=1, b:=2, c:=3)"),
    ],
)
def test_miscellaneous(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "value, init, base_type, full_type",
    [
        param("fValue : INT;", tf.TypeInitialization, "INT", "INT"),
        param("fValue : INT (0..10);", tf.SubrangeTypeInitialization, "INT", "INT (0..10)"),
        param("fValue : (A, B);", tf.EnumeratedTypeInitialization, "INT", "INT"),
        param("fValue : (A, B) DINT;", tf.EnumeratedTypeInitialization, "DINT", "DINT"),
        param(
            "fValue : ARRAY [1..10] OF INT;",
            tf.ArrayTypeInitialization,
            "INT",
            "ARRAY [1..10] OF INT",
        ),
        param(
            "fValue : FB_Test(1, 2, 3);",
            tf.InitializedStructure,
            "FB_Test",
            "FB_Test",
            marks=pytest.mark.xfail(reason="Overlap with function block invocation")
        ),
        param(
            "fValue : FB_Test(A := 1, B := 2, C => 3);",
            tf.FunctionCall,
            "FB_Test",
            "FB_Test",
        ),

        # Aliased by TypeInitialization, it has been removed from the grammar:
        # param("fValue : fbName;", lark.Token, "fbName", "fbName"),
        # Aliased by TypeInitialization:
        param(
            "fValue : STRING[10] := 'abc';",
            tf.StringTypeSpecification,
            "STRING",
            "STRING[10]",
            marks=pytest.mark.xfail(reason="Overlap with TypeInitialization")
        ),
    ]
)
def test_global_types(value, init, base_type, full_type):
    parser = conftest.get_grammar(start="global_var_decl")
    transformed = parse_source_code(value, parser=parser)
    assert isinstance(transformed, tf.GlobalVariableDeclaration)
    assert transformed.variables == ["fValue"]

    assert isinstance(transformed.init, init)

    assert transformed.spec.variables
    assert transformed.base_type_name == base_type
    assert transformed.full_type_name == full_type


@pytest.mark.parametrize(
    "code, comments, pragmas",
    [
        param(
            """
            FUNCTION_BLOCK test
            END_FUNCTION_BLOCK
            """,
            [],
            [],
            id="no_comments",
        ),
        param(
            """
            (* Comment *)
            FUNCTION_BLOCK test
            END_FUNCTION_BLOCK
            """,
            ["(* Comment *)"],
            [],
            id="multiline_comment",
        ),
        param(
            """
            // Comment
            FUNCTION_BLOCK test
            END_FUNCTION_BLOCK
            """,
            ["// Comment"],
            [],
            id="single_line_comment",
        ),
        param(
            """
            // Comment
            (* Comment *)
            FUNCTION_BLOCK test
            END_FUNCTION_BLOCK
            """,
            ["// Comment", "(* Comment *)"],
            [],
            id="both_comments",
        ),
        param(
            """
            // Comment
            (* Comment *)
            {pragma}
            FUNCTION_BLOCK test
            END_FUNCTION_BLOCK
            """,
            ["// Comment", "(* Comment *)"],
            ["{pragma}"],
            id="both_comments_and_pragma1",
        ),
        param(
            """
            // Comment
            {pragma}
            (* Comment *)
            FUNCTION_BLOCK test
            END_FUNCTION_BLOCK
            """,
            ["// Comment", "(* Comment *)"],
            ["{pragma}"],
            id="both_comments_and_pragma2",
        ),
    ]
)
def test_meta(code: str, comments: List[str], pragmas: List[str]):
    transformed = parse_source_code(code)
    meta = transformed.items[0].meta
    found_comments, found_pragmas = meta.get_comments_and_pragmas()
    assert [str(comment) for comment in found_comments] == comments
    assert [str(pragma) for pragma in found_pragmas] == pragmas


@pytest.mark.parametrize(
    "statement, cls",
    [
        ("CONTINUE;", tf.ContinueStatement),
        ("EXIT;", tf.ExitStatement),
        ("RETURN;", tf.ReturnStatement),
    ]
)
def test_statement_priority(statement: str, cls: type):
    transformed = roundtrip_rule("statement_list", statement)
    assert isinstance(transformed, tf.StatementList)
    transformed_statement, = transformed.statements
    assert isinstance(transformed_statement, cls)
