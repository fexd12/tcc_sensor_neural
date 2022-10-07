/* Brain.i */
%module brain
%include typemaps.i

%{
    #define SWIG_FILE_WITH_INIT
    #include "Brain.h"
%}

/* Let's just grab the original header file here */
%include "Brain.h"
