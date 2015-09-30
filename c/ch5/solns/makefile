include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

cartoon: cartoon.o chkopts
	-${CLINKER} -o cartoon cartoon.o ${PETSC_SNES_LIB}
	${RM} cartoon.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ cartoon

