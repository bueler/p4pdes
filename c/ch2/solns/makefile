include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vmksparse: vmksparse.o  chkopts
	-${CLINKER} -o vmksparse vmksparse.o  ${PETSC_KSP_LIB}
	${RM} vmksparse.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ vmksparse

