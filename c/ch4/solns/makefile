include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# solutions to exercises

atan: atan.o chkopts
	-${CLINKER} -o atan atan.o ${PETSC_LIB}
	${RM} atan.o

bratu1D: bratu1D.o chkopts
	-${CLINKER} -o bratu1D bratu1D.o ${PETSC_LIB}
	${RM} bratu1D.o

ecdigits: ecdigits.o chkopts
	-${CLINKER} -o ecdigits ecdigits.o ${PETSC_LIB}
	${RM} ecdigits.o

census: census.o chkopts
	-${CLINKER} -o census census.o ${PETSC_LIB}
	${RM} census.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ atan bratu1D ecdigits census

