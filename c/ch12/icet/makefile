include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

icet: icet.o  chkopts
	-${CLINKER} -o icet icet.o  ${PETSC_LIB}
	${RM} icet.o

# testing

runicet_1:
	-@../../testit.sh icet "-da_refine 2 -ice_verif 1 -ice_eps 0.0 -ice_tf 40 -ice_dt_init 10 -ts_type beuler -ts_adapt_type none" 1 1

runicet_2:
	-@../../testit.sh icet "-da_refine 2 -ice_verif 2 -ice_eps 0.0 -ice_tf 10 -ice_dt_init 3 -snes_fd_color" 2 2

runicet_3:
	-@../../testit.sh icet "-da_refine 2 -ice_tf 4 -ice_dt_init 1 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type none" 1 3

runicet_4:
	-@../../testit.sh icet "-da_refine 2 -ice_tf 20 -ice_dt_init 1 -snes_converged_reason -ts_adapt_clip 0.1,10.0" 1 4

test_icet: runicet_1 runicet_2 runicet_3 runicet_4

test: test_icet

# etc

.PHONY: distclean runicet_1 runicet_2 runicet_3 runicet_4 test test_icet

distclean:
	@rm -f *~ icet *.pyc *tmp
	@rm -f PetscBinaryIO.py petsc_conf.py

