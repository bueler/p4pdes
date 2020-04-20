# testing

# NOTE I expect to remove "-vectorlap" as things get updated ... it is put here so tests pass using the old code

runstokes_1:
	-@../../c/testit.sh stokes.py "-analytical -mx 4 -my 4 -refine 1 -s_ksp_type minres -s_ksp_converged_reason -s_pc_type none" 1 1

runstokes_2:
	-@../../c/testit.sh stokes.py "-udegree 3 -pdegree 2 -quad -showinfo -s_ksp_type preonly -s_pc_type lu -s_pc_factor_shift_type inblocks -s_pmat_type aij" 1 2

runstokes_3:
	-@./lidbox.py -quiet -cl 0.4 -cornerrefine 4 coarse.geo
	-@gmsh -v 0 -2 coarse.geo  # verbosity=0 for quiet
	-@../../c/testit.sh stokes.py "-mesh coarse.msh -refine 1 -dm_view -s_ksp_type fgmres -s_ksp_converged_reason -schurgmg lower -schurpre selfp" 1 3

runstokes_4:
	-@../../c/testit.sh stokes.py "-pdegree 0 -dp -refine 1 -s_ksp_type minres -s_ksp_converged_reason -schurgmg diag -schurpre mass" 1 4

runstokes_5:
	-@../../c/testit.sh stokes.py "-nobase -refine 1 -dm_view -showinfo -s_ksp_monitor_short -s_ksp_rtol 1.0e-3 -schurgmg lower" 2 5

runstokes_6:
	-@../../c/testit.sh stokes.py "-vectorlap -analytical -refine 1 -s_ksp_type gmres -s_ksp_converged_reason -schurgmg lower" 1 6

test_stokes: runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 runstokes_6

test: test_stokes

# etc

.PHONY: clean runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 runstokes_6 test_stokes test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu *.m maketmp tmp difftmp

