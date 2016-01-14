include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

shuffle: shuffle.o  chkopts
	-${CLINKER} -o shuffle shuffle.o  ${PETSC_LIB}
	${RM} shuffle.o

lntwo: lntwo.o  chkopts
	-${CLINKER} -o lntwo lntwo.o  ${PETSC_LIB}
	${RM} lntwo.o

balanced: balanced.o  chkopts
	-${CLINKER} -o balanced balanced.o  ${PETSC_LIB}
	${RM} balanced.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ shuffle lntwo balanced

