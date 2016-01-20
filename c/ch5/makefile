include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

plap: plap.o chkopts
	-${CLINKER} -o plap plap.o ${PETSC_SNES_LIB}
	${RM} plap.o

# testing
runplap_1:
	-@../testit.sh plap "-snes_fd -snes_fd_function" 1 1

runplap_2:
	-@../testit.sh plap "-snes_fd_color" 1 2

runplap_3:
	-@../testit.sh plap "-snes_fd_color -snes_converged_reason -plap_p 1.5 -plap_alpha 0.5 -plap_eps 1.0e-2" 1 3

runplap_4:
	-@../testit.sh plap "-snes_fd_color -snes_converged_reason -da_refine 2 -snes_linesearch_type cp" 4 4

runplap_5:
	-@../testit.sh plap "-plap_quaddegree 1" 1 5

runplap_6:
	-@../testit.sh plap "-snes_converged_reason -ksp_type cg -pc_type icc -da_refine 3" 1 6

# etc

.PHONY: distclean

distclean:
	@rm -f *~ plap

