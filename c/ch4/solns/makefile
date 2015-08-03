include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# solutions to exercises

atan: atan.o chkopts
	-${CLINKER} -o atan atan.o ${PETSC_SNES_LIB}
	${RM} atan.o

bratu: bratu.o chkopts
	-${CLINKER} -o bratu bratu.o ${PETSC_SNES_LIB}
	${RM} bratu.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ atan bratu

