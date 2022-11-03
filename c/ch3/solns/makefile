include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

poisson1D: poisson1D.o
	-${CLINKER} -o poisson1D poisson1D.o ${PETSC_LIB}
	${RM} poisson1D.o

# etc

.PHONY: clean distclean

distclean: clean

clean::
	@rm -f *~ poisson1D

