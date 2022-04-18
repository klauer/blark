FUNCTION_BLOCK PUBLIC class_JsonManager
    VAR_OUTPUT
        (*TRUE if the class is unable to parse a JSON data structure or an
    navigational or extraction method fails due to invalid JSON architecture.*)
        Error     : BOOL;
    END_VAR
END_FUNCTION_BLOCK