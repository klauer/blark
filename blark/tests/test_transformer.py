import pathlib
from typing import Optional

import pytest
from pytest import param

from .. import transform as tf
from ..parse import parse_source_code
from .conftest import get_grammar

# try:
#     import apischema
# except ImportError:
#     # apischema is optional for serialization testing
#     apischema = None


TEST_PATH = pathlib.Path(__file__).parent


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
        "minutes",

        # handled as aliases
        "case_list",

        # handled as tree
        "global_var_list",
        "var_body",

    }

    todo_rules = {
        # program configuration
        "prog_cnxn",
        "prog_conf_element",
        "prog_conf_elements",
        "program_configuration",
        "program_var_declarations",
        "fb_task",

        # tasks
        "configuration_declaration",
        "instance_specific_init",
        "instance_specific_initializations",

        # resources
        "resource_declaration",
        "single_resource_declaration",
    }

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
        param("simple_type_declaration", "TypeName EXTENDS a.b : POINTER TO INT"),
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
            marks=pytest.mark.xfail(reason="TODO; this is valid grammar, I think"),
            # Identical paths:
            #   fb_name_decl -> structure_initialization
            #   array_initialization -> array_initial_element -> structure_initialization
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
            marks=pytest.mark.xfail(reason="TODO; this is valid grammar, I think"),
            # Appears to collide with enum rule; need to fix
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
                fbTest : FB_Test := (1, 2, 3);
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
            marks=pytest.mark.xfail(reason="TODO; this is valid grammar, I think"),
            # Appears to collide with enum rule; need to fix
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
        param("global_var_declarations", tf.multiline_code_block(
            """
            VAR_GLOBAL
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
def test_global_roundtrip(rule_name, value):
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
    ],
)
def test_fb_roundtrip(rule_name, value):
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
    _ = roundtrip_rule(rule_name, value)


def roundtrip_rule(rule_name: str, value: str, expected: Optional[str] = None):
    parser = get_grammar(start=rule_name)
    transformed = parse_source_code(value, parser=parser)
    print("\n\nTransformed:")
    print(repr(transformed))
    print("\n\nOr:")
    print(transformed)
    if expected is None:
        expected = value
    assert str(transformed) == expected

    # if apischema is not None:
    #     serialized = apischema.serialize(transformed)
    #     print("serialized", serialized)
    #     deserialized = apischema.deserialize(type(transformed), serialized)
    #     print("deserialized", deserialized)
    #     assert transformed == deserialized
    return transformed


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
    ],
)
def test_data_type_declaration(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("access_path", "resource.%IX1.1"),
        param("access_path", "resource.prog.func1.func2.Variable"),
        param("access_path", "resource.func2.Variable"),
        param(
            "config_access_declaration",
            "AccessName : resource.%IX1.1 : TypeName READ_ONLY"
        ),
        param(
            "config_access_declaration",
            "AccessName : resource.%IX1.1 : TypeName"
        ),
        param(
            "config_access_declaration",
            "AccessName : resource.Variable : TypeName READ_WRITE"
        ),
        param("config_access_declarations", tf.multiline_code_block(
            """
            (* This is an access block *)
            VAR_ACCESS
                (* Access 1 *)
                AccessName1 : resource.Variable : TypeName READ_WRITE;
                (* Access 2 *)
                AccessName2 : resource.Variable : TypeName READ_ONLY;
                (* Access 3 *)
                AccessName3 : resource.Variable : TypeName;
                (* Access 4 *)
                AccessName4 : resource.%IX1.1 : TypeName;
            END_VAR
            """
        )),
        param("task_initialization", "(SINGLE := 1, INTERVAL := 2, PRIORITY := 3)"),
        param("task_initialization", "(INTERVAL := 2, PRIORITY := 3)"),
        param("task_initialization", "(SINGLE := 1, PRIORITY := 3)"),
        param("task_initialization", "(PRIORITY := 3)"),
        param("task_initialization", "(SINGLE := abc.def, PRIORITY := 3)"),
        param("task_configuration", "TASK taskname (PRIORITY := 3)"),
    ],
)
def test_config_roundtrip(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION ILTest : INT
                LD Speed
                GT 2000
                JMPCN VOLTS_OK
                LD Volts
                VOLTS_OK: LD 1
                ST %QX75
            END_FUNCTION
            """
        )),
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION ILTest : INT
                LD LoadVar
                ST toninstance.IN
                CAL fb(1, 2, 3)
                CAL toninstance(
                    PT := t1,
                    ET => tOut2
                )
                LD toninst1.Q
                JMPC labelname
                ST otherton.IN
                labelname: LD iVar2
                SUB 100
            END_FUNCTION
            """
        )),
        # I don't understand IL well enough, but this is apparently valid
        # grammar
        param("function_declaration", tf.multiline_code_block(
            """
            FUNCTION ILTest : INT
                ADD(iOperand
                    LD test
                    ST test1
                )
                ADD(iOperand)
                end: RET
            END_FUNCTION
            """
        )),
        param("function_declaration", tf.multiline_code_block(
            """
            // Comments 0
            FUNCTION ILTest : INT
                // Comments 1
                ADD(iOperand
                    LD test
                    ST test1
                )
                // Comments 2
                ADD(iOperand)
                // Comments 3
                end: RET
            END_FUNCTION
            """
        )),
    ]
)
def test_instruction_list(rule_name, value):
    roundtrip_rule(rule_name, value)


@pytest.mark.parametrize(
    "rule_name, value",
    [
        param("action_qualifier", "N"),
        param("action_qualifier", "D, Variable"),
        param("action_qualifier", "D, TIME#1D"),
        param("action_association", "ActionName()"),
        param("action_association", "ActionName(N)"),
        param("action_association", "ActionName(D, TIME#1D)"),
        param("action_association", "ActionName(D, TIME#1D, IndicatorName)"),
        param("action_association", "ActionName(D, TIME#1D, Name1, Name2^)"),
        param("sfc_initial_step", tf.multiline_code_block(
            """
            INITIAL_STEP StepName :
            END_STEP
            """
        )),
        param("sfc_step", tf.multiline_code_block(
            """
            STEP StepName :
            END_STEP
            """
        )),
        param("sfc_step", tf.multiline_code_block(
            """
            STEP StepName :
                iValue := iValue + 1;
            END_STEP
            """
        )),
        param("sfc_step", tf.multiline_code_block(
            """
            STEP StepName :
                ActionName(D, TIME#1D, Name1, Name2^)
            END_STEP
            """
        )),
        param("sfc_transition", tf.multiline_code_block(
            """
            TRANSITION TransitionName
            FROM StepName1 TO StepName2 := 1
            END_TRANSITION
            """
        )),
        param("sfc_transition", tf.multiline_code_block(
            """
            TRANSITION TransitionName
            FROM (StepName1, StepName2) TO StepName3 := 1
            END_TRANSITION
            """
        )),
    ]
)
def test_sfc_sequential_function_chart(rule_name, value):
    roundtrip_rule(rule_name, value)
