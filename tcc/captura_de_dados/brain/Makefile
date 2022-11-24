INTERFACE  = Brain.i
LIBS       = -lm
IWRAP      = $(INTERFACE:.i=_wrap.i)
ISRCS      = $(IWRAP:.i=.c)
ICXXSRCS   = $(IWRAP:.i=.cxx)

build:
	swig -python -py3 -c++ -o ${ICXXSRCS} ${INTERFACE}
	python setup.py build_ext --inplace 

build_ext:
	python setup.py build_ext --swig-cpp --inplace