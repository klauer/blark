TYPE
    color: (red, green, blue);
END_TYPE

FUNCTION_BLOCK Testblock
    VAR_INPUT
        number_of_boxes: INT;
    END_VAR
    VAR
        cabinet: color;
    END_VAR

    IF number_of_boxes > 1 THEN
        cabinet := green;
    END_IF
END_FUNCTION_BLOCK
