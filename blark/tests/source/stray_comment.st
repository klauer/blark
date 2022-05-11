(* Call this method to truncate an LREAL value *);
{attribute 'hide'}
METHOD prv_Floor : LREAL
VAR_INPUT
        Input : LREAL;
END_VAR
VAR
        Round : LINT;
END_VAR
Round := LREAL_TO_LINT(Input);
IF LINT_TO_LREAL(Round) <= Input THEN
        prv_Floor := LINT_TO_LREAL(Round);
ELSE
        prv_Floor := LINT_TO_LREAL(Round - 1);
END_IF
END_METHOD
