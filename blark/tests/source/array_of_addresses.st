FUNCTION_BLOCK class_SomethingCool
VAR
        currentChannel : ARRAY[1..g_c_someConstant] OF POINTER TO struct_groupData :=
                [       ADR(_object[1].someValue),
                        ADR(_object[2].someValue),
                        ADR(_object[3].someValue),
                        ADR(_object[4].someValue),
                        ADR(_object[5].someValue),
                        ADR(_object[6].someValue),
                        ADR(_object[7].someValue),
                        ADR(_object[8].someValue),
                        ADR(_object[9].someValue),
                        ADR(_object[10].someValue)
                ];
END_VAR
END_FUNCTION_BLOCK
