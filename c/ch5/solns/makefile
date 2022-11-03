include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

stiff: stiff.o
	-${CLINKER} -o stiff stiff.o  ${PETSC_LIB}
	${RM} stiff.o

.PHONY: clean distclean

distclean: clean

clean::
	@rm -f *~ stiff *.pyc *.dat *.dat.info
