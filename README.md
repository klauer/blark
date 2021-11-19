Beckhoff TwinCAT IEC 61131-3 Lark-based Structured Text Tools
=============================================================

Or for short, blark.  B(eckhoff)-lark. It sounded good in my head, at least.

The Grammar
-----------

The [grammar](blark/iec.lark) uses Lark's Earley parser algorithm.

The grammar itself is not perfect.  It may not reliably parse your source code
or produce useful Python instances just yet.

See [issues](https://github.com/klauer/blark/issues) for further details.

The plan
--------

As a fun side project, blark isn't at the top of my priority list.

Once I get around to it, I hope to:

- [x] Introduce user-friendly Python dataclasses for all PLC constructs
- [x] Create a lark Transformer to take tokenized PLC code and map them onto
  those dataclasses
- [ ] Fix the grammar and improve it as I go
- [ ] Python ``black``-style automatic code formatter?
- [ ] Documentation generator in markdown?
- [ ] Syntax highlighted source code output?

Requirements
------------

* [lark](https://github.com/lark-parser/lark) (for grammar-based parsing)
* [pytmc](https://github.com/pcdshub/pytmc) (for directly loading TwinCAT projects)

How to use it
-------------

1. Preferably using non-system Python, set up an environment using, e.g., miniconda:
```bash
$ conda create -n blark-env python=3.7
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

If you want to use the tokenized source for your own purposes, you'll have
to dig into the source code from there.

Acknowledgements
----------------

Originally based on Volker Birk's IEC 61131-3 grammar
[iec2xml](https://fdik.org/iec2xml/) (GitHub fork
[here](https://github.com/klauer/iec2xml)) and [A Syntactic
Specification for the Programming Languages of theIEC 61131-3
Standard](https://www.researchgate.net/publication/228971719_A_syntactic_specification_for_the_programming_languages_of_the_IEC_61131-3_standard)
by Flor Narciso et al.  Many aspects of the grammar have been added to,
modified, and in cases entirely rewritten to better support lark grammars and
transformers.
