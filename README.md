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

* Introduce user-friendly Python dataclasses for all PLC constructs
* Create a lark Transformer to take tokenized PLC code and map them onto those
  dataclasses
* Fix the grammar and improve it as I go

Requirements
------------

* lark-parser
* pytmc (for parsing TwinCAT projects)

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

3. Run the parser.  Supported file types include those from TwinCAT3 projects (
   ``.tsproj`., ``.sln``, ``.TcPOU``, ``.TcGVL``).

```bash
$ blark parse -vvv blark/tests/POUs/F_SetStateParams.TcPOU
start
  iec_source
    function_declaration
      derived_function_name     F_SetStateParams
      bit_string_type_name      BOOL
      input_declarations
        var_input_body
          var_init_decl
            variable_name       nStateRef
            simple_spec_init
              integer_type_name UDINT
...
```

If you want to use the tokenized source for your own purposes, you'll have
to dig into the source code from there.

Acknowledgements
----------------

Based on Volker Birk's IEC 61131-3 grammar [iec2xml](https://fdik.org/iec2xml/)
(GitHub fork [here](https://github.com/klauer/iec2xml)) and [A Syntactic
Specification for the Programming Languages of theIEC 61131-3
Standard](https://www.researchgate.net/publication/228971719_A_syntactic_specification_for_the_programming_languages_of_the_IEC_61131-3_standard)
by Flor Narciso et al.
