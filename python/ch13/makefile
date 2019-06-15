# testing

runfish_1:
	-@../../c/testit.sh fish.py "-mx 5 -my 5 -k 2 -s_ksp_type preonly -s_pc_type lu" 1 1

runfish_2:
	-@../../c/testit.sh fish.py "-refine 2 -s_ksp_converged_reason -s_pc_type mg -s_mg_levels_ksp_max_it 3" 1 2

runfish_3:
	-@../../c/testit.sh fish.py "-refine 1 -s_snes_type newtonls -s_snes_converged_reason -s_ksp_converged_reason -s_pc_type gamg -dm_view" 1 3

runstokes_1:
	-@../../c/testit.sh stokes.py "-mx 4 -my 4 -refine 1 -analytical -s_ksp_converged_reason" 1 1

runstokes_2:
	-@../../c/testit.sh stokes.py "-udegree 3 -pdegree 2 -quad -package directsvd -show_norms" 1 2

runstokes_3:
	-@./lidbox.py -quiet -cl 0.4 -cornerrefine 4 coarse.geo
	-@gmsh -v 0 -2 coarse.geo  # verbosity=0 for quiet
	-@../../c/testit.sh stokes.py "-i coarse.msh -show_norms -dm_view -refine 1 -package schur3 -s_ksp_converged_reason" 1 3

runstokes_4:
	-@../../c/testit.sh stokes.py "-refine 1 -pdegree 0 -dpressure -show_norms -package schur3" 1 4

runstokes_5:
	-@../../c/testit.sh stokes.py "-s_ksp_monitor_short -s_ksp_rtol 1.0e-3 -refine 1 -nobase -show_norms -package schur3" 2 5

test_fish: runfish_1 runfish_2 runfish_3

test_stokes: runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5

test: test_fish test_stokes

# etc

.PHONY: clean runfish_1 runfish_2 runfish_3 runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 test_fish test_stokes test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu *.m maketmp tmp difftmp

