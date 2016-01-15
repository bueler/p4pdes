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


# etc

.PHONY: distclean

distclean:
	@rm -f *~ plap

