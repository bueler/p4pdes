# testing

runfish_1:
	-@../../c/testit.sh fish.py "-mx 5 -my 5 -order 2 -s_ksp_type preonly -s_pc_type lu" 1 1

runfish_2:
	-@../../c/testit.sh fish.py "-refine 2 -s_ksp_converged_reason -s_pc_type mg" 1 2

runstokes_1:
	-@../../c/testit.sh stokes.py "-mx 4 -my 4 -refine 1 -mu 10.0 -show_norms" 1 1

runstokes_2:
	-@../../c/testit.sh stokes.py "-uorder 3 -porder 2 -quad -s_mat_type aij -s_ksp_type preonly -s_pc_type svd -show_norms" 1 2

runstokes_3:
	-@gmsh -v 0 -2 lidbox.geo  # verbosity=0 for quiet
	-@../../c/testit.sh stokes.py "-i lidbox.msh -show_norms" 1 3

runstokes_4:
	-@../../c/testit.sh stokes.py "-porder 0 -dpressure -show_norms" 1 4

test_fish: runfish_1 runfish_2

test_stokes: runstokes_1 runstokes_2 runstokes_3 runstokes_4

test: test_fish test_stokes

# etc

.PHONY: clean runfish_1 runfish_2 runstokes_1 runstokes_2 runstokes_3 runstokes_4 test_fish test_stokes test

clean:
	@rm -f *.pyc *.msh *.pvd *.vtu maketmp tmp difftmp

