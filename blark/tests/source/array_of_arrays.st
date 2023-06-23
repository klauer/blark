{attribute 'hide'}
METHOD prv_Detection : BOOL
VAR_IN_OUT
        currentChannel : ARRAY[APhase..CPhase] OF ARRAY [1..5] OF INT := [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]
        ];
END_VAR

// do some stuff
END_METHOD
