import pathlib

import pytest
from pytest import param

from .. import transform as tf
from .conftest import get_grammar, stringify_tokens

TEST_PATH = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    "name, value, expected",
    [
        param("integer_literal", "-12", tf.Integer(value="-12")),
        param("integer_literal", "12", tf.Integer(value="12")),
        param("integer_literal", "INT#12", tf.Integer(value="12", type="INT")),
        param("integer_literal", "2#10010", tf.BinaryInteger(value="10010")),
        param("integer_literal", "8#22", tf.OctalInteger(value="22")),
        param("integer_literal", "16#12", tf.HexInteger(value="12")),
        param("integer_literal", "UDINT#12", tf.Integer(value="12", type="UDINT")),
        param("integer_literal", "UDINT#2#010", tf.BinaryInteger(value="010", type="UDINT")),  # noqa: E501
        param("integer_literal", "UDINT#2#1001_0011", tf.BinaryInteger(value="1001_0011", type="UDINT")),  # noqa: E501
        param("integer_literal", "DINT#16#C0FFEE", tf.HexInteger(value="C0FFEE", type="DINT")),  # noqa: E501
        param("real_literal", "-12.0", tf.Real(value="-12.0")),
        param("real_literal", "12.0", tf.Real(value="12.0")),
        param("real_literal", "12.0e5", tf.Real(value="12.0e5")),
        param("bit_string_literal", "WORD#1234", tf.BitString(type="WORD", value="1234")),
        param("bit_string_literal", "WORD#2#0101", tf.BinaryBitString(type="WORD", value="0101")),  # noqa: E501
        param("bit_string_literal", "WORD#8#777", tf.OctalBitString(type="WORD", value="777")),  # noqa: E501
        param("bit_string_literal", "word#16#FEEE", tf.HexBitString(type="word", value="FEEE")),  # noqa: E501
        param("boolean_literal", "BOOL#1", tf.Boolean(value="1")),
        param("boolean_literal", "BOOL#0", tf.Boolean(value="0")),
        param("boolean_literal", "BOOL#TRUE", tf.Boolean(value="TRUE")),
        param("boolean_literal", "BOOL#FALSE", tf.Boolean(value="FALSE")),
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
        # pytest.param("single_byte_character_string", "'abc'"),
        # pytest.param("double_byte_character_string", '"abc"'),
        # pytest.param("single_byte_string_spec", "STRING[1]"),
        # pytest.param("single_byte_string_spec", "STRING(1)"),
        # pytest.param("single_byte_string_spec", "STRING(1) := 'abc'"),
        # pytest.param("double_byte_string_spec", "WSTRING[1]"),
        # pytest.param("double_byte_string_spec", "WSTRING(1)"),
        # pytest.param("double_byte_string_spec", 'WSTRING(1) := "abc"'),
    ],
)
def test_literal(name, value, expected):
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    print(f"transformed -> {transformed}")
    assert stringify_tokens(transformed) == expected


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
    ],
)
def test_literal_roundtrip(name, value):
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    assert str(transformed) == value


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
    parsed = get_grammar(start=name).parse(value)
    print(f"rule {name} value {value!r} into {parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    assert str(transformed) == expected


def roundtrip_rule(rule_name: str, value: str):
    parsed = get_grammar(start=rule_name).parse(value)
    print(f"rule {rule_name} value {value!r} into:\n\n{parsed}")
    transformed = tf.GrammarTransformer().transform(parsed)
    print("\n\nTransformed:")
    print(repr(transformed))
    print("\n\nOr:")
    print(transformed)
    assert str(transformed) == value
    return transformed


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
            # Appears to collide with enum rule; need to fix
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
            """
        )),
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
            """
        )),
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
