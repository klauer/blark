(* This program uses some object-oriented function-block with multiline access. *)
PROGRAM oop_test
VAR
        uut : SomeFunctionBlock;
END_VAR

// Use an object that has a different method called on each line
uut.DoSomething(input_1:='foo', input_2:='bar')
   .DoSomethingElse(input_1:=5)
   .Finish();

// Or perhaps something that's a little more tame
uut.DoSomething(input_1:='foo', input_2:='bar')
   .DoSomethingElse(input_1:=5).Finish();

// Or something tamer, still
uut.DoSomething(input_1:='foo', input_2:='bar').DoSomethingElse(input_1:=5).Finish();

END_PROGRAM
