include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

plap: plap.o chkopts
	-${CLINKER} -o plap plap.o ${PETSC_SNES_LIB}
	${RM} plap.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ plap

