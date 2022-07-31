include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vmkrankzero: vmkrankzero.o
	-${CLINKER} -o vmkrankzero vmkrankzero.o  ${PETSC_KSP_LIB}
	${RM} vmkrankzero.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ vmkrankzero

