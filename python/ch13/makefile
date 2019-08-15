# testing

runfish_1:
	-@../../c/testit.sh fish.py "-mx 5 -my 5 -k 2 -s_ksp_type preonly -s_pc_type lu" 1 1

runfish_2:
	-@../../c/testit.sh fish.py "-refine 2 -dm_view -s_ksp_converged_reason -s_pc_type mg -s_mg_levels_ksp_max_it 3" 2 2

runfish_3:
	-@../../c/testit.sh fish.py "-refine 1 -dm_view -s_snes_type newtonls -s_snes_converged_reason -s_ksp_converged_reason -s_pc_type gamg" 1 3

runfish_4:
	-@../../c/testit.sh fish.py "-fishhelp" 1 4

runstokes_1:
	-@../../c/testit.sh stokes.py "-mx 4 -my 4 -refine 1 -s_ksp_type minres -s_pc_type none -analytical -s_ksp_converged_reason" 1 1

runstokes_2:
	-@../../c/testit.sh stokes.py "-showinfo -udegree 3 -pdegree 2 -quad -s_ksp_type preonly -s_pc_type lu -s_pc_factor_shift_type inblocks -s_pmat_type aij" 1 2

runstokes_3:
	-@./lidbox.py -quiet -cl 0.4 -cornerrefine 4 coarse.geo
	-@gmsh -v 0 -2 coarse.geo  # verbosity=0 for quiet
	-@../../c/testit.sh stokes.py "-mesh coarse.msh -dm_view -refine 1 -s_ksp_type fgmres -s_ksp_converged_reason -schurgmg lower" 1 3

runstokes_4:
	-@../../c/testit.sh stokes.py "-refine 1 -pdegree 0 -dp -s_ksp_type minres -s_ksp_converged_reason -schurgmg diag_mass" 1 4

runstokes_5:
	-@../../c/testit.sh stokes.py "-nobase -refine 1 -dm_view -showinfo -s_ksp_monitor_short -s_ksp_rtol 1.0e-3 -schurgmg lower" 2 5

runstokes_6:
	-@../../c/testit.sh stokes.py "-stokeshelp" 1 6

test_fish: runfish_1 runfish_2 runfish_3 runfish_4

test_stokes: runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 runstokes_6

test: test_fish test_stokes

# etc

.PHONY: clean runfish_1 runfish_2 runfish_3 runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 test_fish test_stokes test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu *.m maketmp tmp difftmp

