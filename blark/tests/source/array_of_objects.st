{attribute 'hide'}
METHOD prv_Detection : BOOL
VAR_IN_OUT
        currentChannel : ARRAY[APhase..CPhase] OF class_baseVector(SIZEOF(vector_t),0);
END_VAR

// do some stuff
END_METHOD
