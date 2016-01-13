include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

fish2: fish2.o chkopts
	-${CLINKER} -o fish2 fish2.o ${PETSC_SNES_LIB}
	${RM} fish2.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ fish2

