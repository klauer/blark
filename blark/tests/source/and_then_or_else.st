(*<textarea rows="15" cols="81">
********************************************************************************
**** a great, big, long multiline comment
********************************************************************************
</textarea>*)
METHOD Split : POINTER TO STRING (*Returns a pointer to an error string.
Returns zero on success.*)
VAR_IN_OUT
    (* The class_SELString containing the combination of characters to denote 
the locations where the split shall occur.*)
    sep    : class_Object;
END_VAR

IF TestOne AND_THEN TestTwo THEN
    DoSomething();
ELSIF TestThree OR_ELSE TestFour THEN
    DoSomethingElse();
END_IF

END_METHOD