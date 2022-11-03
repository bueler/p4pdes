include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

vmkrankzero: vmkrankzero.o
	-${CLINKER} -o vmkrankzero vmkrankzero.o  ${PETSC_LIB}
	${RM} vmkrankzero.o

.PHONY: clean distclean

distclean: clean

clean::
	@rm -f *~ vmkrankzero

