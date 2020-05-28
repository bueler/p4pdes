include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

expx: expx.o
	-${CLINKER} -o expx expx.o  ${PETSC_LIB}
	${RM} expx.o

balanced: balanced.o
	-${CLINKER} -o balanced balanced.o  ${PETSC_LIB}
	${RM} balanced.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ expx balanced

