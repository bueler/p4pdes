include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

atan: atan.o
	-${CLINKER} -o atan atan.o ${PETSC_LIB}
	${RM} atan.o

bratu1D: bratu1D.o
	-${CLINKER} -o bratu1D bratu1D.o ${PETSC_LIB}
	${RM} bratu1D.o

ecdigits: ecdigits.o
	-${CLINKER} -o ecdigits ecdigits.o ${PETSC_LIB}
	${RM} ecdigits.o

census: census.o
	-${CLINKER} -o census census.o ${PETSC_LIB}
	${RM} census.o

.PHONY: clean distclean

distclean:

clean::
	@rm -f *~ atan bratu1D ecdigits census

