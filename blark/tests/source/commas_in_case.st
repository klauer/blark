(*<textarea rows="15" cols="81">
********************************************************************************
* Comments
********************************************************************************
</textarea>*)
FUNCTION fun_CommainCase : BOOL
VAR_INPUT
    (* Byte to be evaluated as a printable character. *)
    character : BYTE;
END_VAR

CASE character OF
    (*
    Any character that falls in the ascii range (in base-10)
    0-8, 11, 12, 14-27 or >=127 is an invalid character.
    *)
    BYTE#9..BYTE#10, BYTE#13, BYTE#28..BYTE#126:
        fun_CommainCase := TRUE;
ELSE
    fun_CommainCase := FALSE;
END_CASE
END_FUNCTION
