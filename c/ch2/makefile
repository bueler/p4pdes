include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


vecmatksp: vecmatksp.o  chkopts
	-${CLINKER} -o vecmatksp vecmatksp.o  ${PETSC_KSP_LIB}
	${RM} vecmatksp.o

tri: tri.o  chkopts
	-${CLINKER} -o tri tri.o  ${PETSC_KSP_LIB}
	${RM} tri.o


# test codes

testmatcreate: testmatcreate.o  chkopts
	-${CLINKER} -o testmatcreate testmatcreate.o  ${PETSC_KSP_LIB}
	${RM} testmatcreate.o


# exercise solutions

e1matvecn: e1matvecn.o  chkopts
	-${CLINKER} -o e1matvecn e1matvecn.o  ${PETSC_KSP_LIB}
	${RM} e1matvecn.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ vecmatksp tri e1matvecn

