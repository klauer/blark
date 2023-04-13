(* This program creates a function block instance with initialization. *)
PROGRAM oop_test
VAR
    fbSample  : FB_Sample(nInitParam := 1) := (nInput := 2, nMyProperty := 3);
END_VAR

END_PROGRAM
