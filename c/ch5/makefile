include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

fish2: fish2.o ../ch3/structuredpoisson.o  chkopts
	-${CLINKER} -o fish2 fish2.o ../ch3/structuredpoisson.o  ${PETSC_KSP_LIB}
	${RM} fish2.o ../ch3/structuredpoisson.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ fish2

