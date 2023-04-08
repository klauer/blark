(* This program creates an array of objects using array initialization. *)
PROGRAM oop_test
VAR
    runners : ARRAY[RunnerTasks.RUNNER_START+1..RunnerTasks.RUNNER_END-1] OF FB_Runner[
        (instanceName := 'one'),
        (instanceName := 'two'),
        (instanceName := 'three'),
        (instanceName := 'four'),
        (instanceName := 'five'),
        (instanceName := 'six'),
        (instanceName := 'seven'),
        (instanceName := 'eight'),
        (instanceName := 'nine'),
        (instanceName := 'ten'),
        (instanceName := 'elevin'),
        (instanceName := 'thirteen'),
        (instanceName := 'fourteen'),
        (instanceName := 'fifteen'),
        (instanceName := 'sixteen'),
        (instanceName := 'seventeen'),
        (instanceName := 'eighteen')
    ];
    aSample   : ARRAY[1..2] OF FB_Sample[(nInitParam := 4), (nInitParam := 7)]
                            := [(nInput := 5, nMyProperty := 6), (nInput := 8, nMyProperty := 9)];
END_VAR

END_PROGRAM
