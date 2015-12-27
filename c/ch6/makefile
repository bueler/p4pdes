include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ad3: ad3.o  chkopts
	-${CLINKER} -o ad3 ad3.o  ${PETSC_SNES_LIB}
	${RM} ad3.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ ad3

