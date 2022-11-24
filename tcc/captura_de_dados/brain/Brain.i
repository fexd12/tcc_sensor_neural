/* Brain.i */
%module brain
%include typemaps.i
%include "carrays.i"
%array_functions(uint32_t, uint32_tArray);

%inline %{
    typedef unsigned char uint8_t;
    #define SWIG_FILE_WITH_INIT
    #include "Brain.h"
%}

/* Let's just grab the original header file here */
%include "Brain.h"
