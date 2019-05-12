include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ad3: ad3.o  chkopts
	-${CLINKER} -o ad3 ad3.o  ${PETSC_LIB}
	${RM} ad3.o

# testing

# defaults: problem = layer, eps = 1.0, centered fluxes, -snes_fd_color
runad3_1:
	-@../../testit.sh ad3 "" 1 1

# problem = layer, eps = 0.01 and upwind and GAMG
runad3_2:
	-@../../testit.sh ad3 "-ad3_eps 0.01 -ad3_limiter none -pc_type gamg" 1 2

# diffusion-only problem with two-level GMG
runad3_3:
	-@../../testit.sh ad3 "-ad3_problem nowind -ad3_limiter none -da_refine 1 -ksp_converged_reason -ksp_type cg -pc_type mg" 1 3


test_ad3: runad3_1 runad3_2 runad3_3

test: test_ad3

# etc

.PHONY: distclean rrunad3_1 runad3_2 runad3_3 test_ad3 test

distclean:
	@rm -f *~ *tmp *.pyc ad3

