{attribute 'qualified_only'}
{attribute 'to_string'}
TYPE color :
(
   off := ANY_TO_DINT(someColor.off),
   red := ANY_TO_DINT(someColor.red),
   green := ANY_TO_DINT(someColor.green)
) DINT;
END_TYPE