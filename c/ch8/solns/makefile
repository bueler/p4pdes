include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

lntwo: lntwo.o  chkopts
	-${CLINKER} -o lntwo lntwo.o  ${PETSC_LIB}
	${RM} lntwo.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ lntwo

