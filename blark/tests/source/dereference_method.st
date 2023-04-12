METHOD DoSomething
VAR
    Something : BOOL;
END_VAR

Something := anObject.WriteLine(THIS^.something.t, THIS^.GetTagName()^, INT_TO_STRING(BOOL_TO_INT(THIS^.someAttribute)));

END_METHOD