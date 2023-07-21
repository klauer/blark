# Beckhoff TwinCAT IEC 61131-3 Lark-based Structured Text Tools

Or for short, blark.  B(eckhoff)-lark. It sounded good in my head, at least.

## The Grammar

The [grammar](blark/iec.lark) uses Lark's Earley parser algorithm.

The grammar itself is not perfect.  It may not reliably parse your source code
or produce useful Python instances just yet.

See [issues](https://github.com/klauer/blark/issues) for further details.

As a fun side project, blark isn't at the top of my priority list.  For
an idea of where the project is going, see the issues list.

## Requirements

* [lark](https://github.com/lark-parser/lark) (for grammar-based parsing)
* [lxml](https://github.com/lxml/lxml) (for parsing TwinCAT projects)

## Capabilities

* TwinCAT source code file parsing (``*.TcPOU`` and others)
* TwinCAT project and solution loading
* ``lark.Tree`` generation of any supported source code
* Python dataclasses of supported source code, with introspection and code refactoring

### Works-in-progress

* Sphinx API documentation generation (a new Sphinx domain)
* Code reformatting
* "Dependency store" - recursively parse and inspect project dependencies
* Summary generation - a layer on top of dataclasses to summarize source code details
* Rewriting source code directly in TwinCAT source code files

## Installation

Installation is quick with Pip.

```bash
pip install --upgrade blark
```

### Quickstart (pip / virtualenv with venv)

1. Set up an environment using venv:
  ```bash
  $ python -m venv blark_venv
  $ source blark_venv/bin/activate
  ```
2. Install the library with pip:
  ```bash
  $ python -m pip install blark
  ```

### Quickstart (Conda)

1. Set up an environment using conda:
  ```bash
  $ conda create -n blark-env -c conda-forge python=3.10 pip blark
  $ conda activate blark-env
  ```
2. Install the library from conda:
  ```bash
  $ conda install blark
  ```

### Development install

If you run into issues or wish to run an unreleased version of blark, you may
install directly from this repository like so:
```bash
$ python -m pip install git+https://github.com/klauer/blark
```

## Sample runs

Run the parser or experimental formatter utility.  Current supported file types
include those from TwinCAT3 projects ( ``.tsproj``, ``.sln``, ``.TcPOU``,
``.TcGVL``) and plain-text ``.st`` files.

```bash
$ blark parse --print-tree blark/tests/POUs/F_SetStateParams.TcPOU
function_declaration
  None
  F_SetStateParams
  indirect_simple_specification
    None
    simple_specification        BOOL
  input_declarations
    None
    var1_init_decl
      var1_list
... (clipped) ...
```

To interact with the Python dataclasses directly, make sure IPython is
installed first and then try:

```
$ blark parse --interactive blark/tests/POUs/F_SetStateParams.TcPOU
# Assuming IPython is installed, the following prompt will come up:

In [1]: results[0].identifier
Out[1]: 'F_SetStateParams/declaration'

In [2]: results[1].identifier
Out[2]: 'F_SetStateParams/implementation'
```

Dump out a parsed and reformatted set of source code:

```bash
$ blark format blark/tests/source/array_of_objects.st
{attribute 'hide'}
METHOD prv_Detection : BOOL
    VAR_IN_OUT
        currentChannel : ARRAY [APhase..CPhase] OF class_baseVector(SIZEOF(vector_t), 0);
    END_VAR
END_METHOD
```

blark supports rewriting TwinCAT source code files directly as well:

```bash
$ blark format blark/tests/POUs/F_SetStateParams.TcPOU

<TcPlcObject Version="1.1.0.1" ProductVersion="3.1.4024.0">
  <POU Name="F_SetStateParams" Id="{f9611d23-4bb5-422d-9f11-2cc94e61fc9e}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION F_SetStateParams : BOOL
    VAR_INPUT
        nStateRef : UDINT;
        rPosition : REAL;
        rTolerance : REAL;
        stBeamParams : ST_BeamParams;

... (clipped) ...
```

It is also possible to parse the source code into a tokenized ``SourceCode``
tree which supports code introspection and rewriting:

```python
In [1]: import blark

In [2]: parsed = blark.parse_source_code(
   ...:     """
   ...: PROGRAM ProgramName
   ...:     VAR_INPUT
   ...:         iValue : INT;
   ...:     END_VAR
   ...:     VAR_ACCESS
   ...:         AccessName : SymbolicVariable : TypeName READ_WRITE;
   ...:     END_VAR
   ...:     iValue := iValue + 1;
   ...: END_PROGRAM
   ...: """
   ...: )

# Access the lark Tree here:
In [3]: parsed.tree.data
Out[3]: Token('RULE', 'iec_source')

# Or the transformed information:
In [3]: transformed = parsed.transform()

In [4]: program = transformed.items[0]

In [5]: program.declarations[0].items[0].variables[0].name
Out[5]: Token('IDENTIFIER', 'iValue')
```

The supported starting grammar rules for the reusable parser include:

```
"iec_source"
"action"
"data_type_declaration"
"function_block_method_declaration"
"function_block_property_declaration"
"function_block_type_declaration"
"function_declaration"
"global_var_declarations"
"program_declaration"
"statement_list"
```

Other starting rules remain possible for advanced users, however a new parser
must be created in that scenario and transformations are not supported.

Additionally, please note that you should avoid creating parsers on-the-fly as
there is a startup cost to re-parsing the grammar. Utilize the provided parser
from ``blark.get_parser()`` whenever possible.

```
In [1]: import blark

In [2]: parser = blark.new_parser(start=["any_integer"])

In [3]: Tree('hex_integer', [Token('HEX_STRING', '1010')])
```

## Adding Test Cases

Presently, test cases are provided in two forms. Within the `blark/tests/`
directory there are `POUs/` and `source/` directories.

TwinCAT source code files belong in ``blark/tests/POUs``.
Plain-text source code files (e.g., ``.st`` files) belong in
``blark/tests/source``.

Feel free to contribute your own test cases and we'll do our best to ensure
that blark parses them (and continues to parse them) without issue.

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
