# Beckhoff TwinCAT IEC 61131-3 Lark-based Structured Text Tools

Or for short, blark.  B(eckhoff)-lark. It sounded good in my head, at least.

## The Grammar

The [grammar](blark/iec.lark) uses Lark's Earley parser algorithm.

The grammar itself is not perfect.  It may not reliably parse your source code
or produce useful Python instances just yet.

See [issues](https://github.com/klauer/blark/issues) for further details.

## The plan

As a fun side project, blark isn't at the top of my priority list.

Once I get around to it, I hope to:

- [x] Introduce user-friendly Python dataclasses for all PLC constructs
- [x] Create a lark Transformer to take tokenized PLC code and map them onto
  those dataclasses
- [ ] Fix the grammar and improve it as I go
- [ ] Python ``black``-style automatic code formatter?
- [ ] Documentation generator in markdown?
- [ ] Syntax highlighted source code output?

## Requirements

- [lark](https://github.com/lark-parser/lark) (for grammar-based parsing)
- [pytmc](https://github.com/pcdshub/pytmc) (for directly loading TwinCAT projects)

## Installation

Installation is quick with Pip.

```bash
pip install --upgrade blark
```

## Quickstart

1. Preferably using non-system Python, set up an environment using, e.g., miniconda:
  ```bash
  $ conda create -n blark-env -c conda-forge python=3.7 blark
  $ conda activate blark-env
  ```
2. Install the library (using conda or otherwise, these steps are the same)
  ```bash
  $ pip install blark
  ```
3. Run the parser or experimental formatter utility.  Supported file types
   include those from TwinCAT3 projects ( ``.tsproj``, ``.sln``, ``.TcPOU``,
   ``.TcGVL``).

```bash
$ blark parse -vvv blark/tests/POUs/F_SetStateParams.TcPOU
iec_source
  function_declaration
    F_SetStateParams
    BOOL
    function_var_blocks
      input_declarations
        None
        var1_init_decl
          var1_list
            var1
              variable_name
                nStateRef
                None
              None
          simple_spec_init
            None
            UDINT
            None
... (clipped) ...
```

To interact with the Python dataclasses directly, use:

```
$ blark parse --interactive blark/tests/POUs/F_SetStateParams.TcPOU
# Assuming IPython is installed, the following prompt will come up:

In [1]: result.items[0].name
Out[1]: Token('IDENTIFIER', 'F_SetStateParams')
```

Dump out a parsed and reformatted set of source code:

```bash
$ blark format blark/tests/POUs/F_SetStateParams.TcPOU
FUNCTION F_SetStateParams : BOOL
    VAR_INPUT
        nStateRef : UDINT;
        rPosition : REAL;
        rTolerance : REAL;
        stBeamParams : ST_BeamParams;
    END_VAR
    VAR_IN_OUT
        Table : FB_LinearDeviceStateTable;
    END_VAR
    VAR
        stDeviceState : ST_DeviceState;
    END_VAR
    stDeviceState.nStateRef := nStateRef;
    stDeviceState.rPosition := rPosition;
    stDeviceState.rTolerance := rTolerance;
    stDeviceState.stReqBeamParam := stBeamParams;
    Table.A_Add(key := nStateRef, putValue := stDeviceState);
    F_SetStateParams := Table.bOk;
END_FUNCTION
```

It is also possible to parse the source code into a tokenized SourceCode tree:

```python
>>> import blark
>>> blark.parse_source_code(
...     """
...                 PROGRAM ProgramName
...                 VAR_INPUT
...                     iValue : INT;
...                 END_VAR
...                 VAR_ACCESS
...                     AccessName : SymbolicVariable : TypeName READ_WRITE;
...                 END_VAR
...                 iValue := iValue + 1;
...             END_PROGRAM
... """
... )
SourceCode(items=[Program(name=Token('IDENTIFIER', 'ProgramName'), declarations=[InputDeclarations(attrs=None, items=[VariableOneInitDeclaration(variables=[DeclaredVariable(variable=SimpleVariable(name=Token('IDENTIFIER', 'iValue'), dereferenced=False), location=None)], init=TypeInitialization(indirection=None, spec=SimpleSpecification(type=Token('DOTTED_IDENTIFIER', 'INT')), value=None))]), AccessDeclarations(items=[AccessDeclaration(name=Token('IDENTIFIER', 'AccessName'), variable=SimpleVariable(name=Token('IDENTIFIER', 'SymbolicVariable'), dereferenced=False), type=DataType(indirection=None, type_name=Token('DOTTED_IDENTIFIER', 'TypeName')), direction=Token('READ_WRITE', 'READ_WRITE'))])], body=StatementList(statements=[AssignmentStatement(variables=[SimpleVariable(name=Token('IDENTIFIER', 'iValue'), dereferenced=False)], expression=BinaryOperation(left=SimpleVariable(name=Token('IDENTIFIER', 'iValue'), dereferenced=False), op=Token('ADD_OPERATOR', '+'), right=Integer(value=Token('INTEGER', '1'), type_name=None)))]))], filename=PosixPath('unknown'), raw_source='\n                PROGRAM ProgramName\n                VAR_INPUT\n                    iValue : INT;\n                END_VAR\n                VAR_ACCESS\n                    AccessName : SymbolicVariable : TypeName READ_WRITE;\n                END_VAR\n                iValue := iValue + 1;\n            END_PROGRAM\n')
```

Alternatively, if you only want the tree:

```python
In [1]: import blark

In [2]: parser = blark.parse.new_parser(start="function_block_body")

In [3]: parser.parse(
    ...:     """// Default return value to TRUE
    ...: SerializeJson := TRUE;
    ...:
    ...: // Set to Root of Structure
    ...: Root();
    ...:
    ...: SerializeJson := SerializeJson AND _serializedContent.Recycle();
    ...:
    ...: // Set up Initial States for Indices
    ...: lastLevel := Current.LEVEL;
    ...: outerType := Current.JSON_TYPE;
    ...: """
    ...: )
Out[3]: Tree(...)
```

For some additional reference regarding this syntax, refer to
[the comment here on issue #20](https://github.com/klauer/blark/issues/20#issuecomment-1099699641)

## Adding Test Cases

Presently, test cases are provided in two forms. Within the `blark/tests/`
directory there are `POUs/` and `source/` directories.

## Acknowledgements

Originally based on Volker Birk's IEC 61131-3 grammar
[iec2xml](https://fdik.org/iec2xml/) (GitHub fork
[here](https://github.com/klauer/iec2xml)) and [A Syntactic
Specification for the Programming Languages of theIEC 61131-3
Standard](https://www.researchgate.net/publication/228971719_A_syntactic_specification_for_the_programming_languages_of_the_IEC_61131-3_standard)
by Flor Narciso et al.  Many aspects of the grammar have been added to,
modified, and in cases entirely rewritten to better support lark grammars and
transformers.

Special thanks to the blark contributors:

- @engineerjoe440

## Related, Similar, or Alternative Projects

There are a number of similar, or related projects that are available.

- ["MATIEC"](https://github.com/nucleron/matiec) - another IEC 61131-3 Structured
Text parser which supports IEC 61131-3 second edition, without classes,
namespaces and other fancy features. An updated version is also
[available on Github](https://github.com/sm1820/matiec)
- [OpenPLC Runtime Version 3](https://github.com/thiagoralves/OpenPLC_v3) -
As stated by the project:
  > OpenPLC is an open-source Programmable Logic Controller that is based on easy to use software. Our focus is to provide a low cost industrial solution for automation and research. OpenPLC has been used in many research papers as a framework for industrial cyber security research, given that it is the only controller to provide the entire source code.
- [RuSTy](https://github.com/PLC-lang/rusty)
[documentation](https://plc-lang.github.io/rusty/intro_1.html) - Structured text
compiler written in Rust. As stated by the project:
  > RuSTy is a structured text (ST) compiler written in Rust. RuSTy utilizes the LLVM framework to compile eventually to native code.
- [IEC Checker](https://github.com/jubnzv/iec-checker) - Static analysis tool
for IEC 61131-3 logic. As described by the maintainer:
  > iec-checker has the ability to parse ST source code and dump AST and CFG to JSON format, so you can process it with your language of choice.
- [TcBlack](https://github.com/Roald87/TcBlack) - Python black-like code formatter for TwinCAT code.
