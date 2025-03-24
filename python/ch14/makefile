# testing

GMSH = gmsh

rungmshversion_1:
	-@../ftestit.sh ${GMSH} "-version" INFO 1

runstokes_1:
	-@../ftestit.sh stokes.py "-analytical -mx 4 -my 4 -s_ksp_type minres -s_ksp_converged_reason -s_pc_type none -s_ksp_rtol 1.0e-2" 1 1

runstokes_2:
	-@../ftestit.sh stokes.py "-udegree 3 -pdegree 2 -refine 1 -quad -showinfo -s_ksp_type preonly -s_pc_type lu -s_pc_factor_shift_type inblocks -s_ksp_monitor_short" 1 2

# following test depends on gmsh version
runstokes_3:
	-@./lidbox.py -quiet -cl 0.4 -cornerrefine 4 coarse.geo
	-@${GMSH} -v 0 -2 coarse.geo
	-@../ftestit.sh stokes.py "-mesh coarse.msh -refine 1 -s_ksp_type fgmres -s_ksp_converged_reason -schurgmg lower -schurpre selfp" 1 3

runstokes_4:
	-@../ftestit.sh stokes.py "-pdegree 0 -dp -refine 1 -s_ksp_type minres -s_ksp_converged_reason -schurgmg diag -schurpre mass" 1 4

runstokes_5:
	-@../ftestit.sh stokes.py "-nobase -refine 1 -showinfo -s_ksp_monitor_short -s_ksp_rtol 1.0e-3 -schurgmg lower" 2 5

runstokes_6:
	-@../ftestit.sh stokes.py "-vectorlap -analytical -refine 1 -s_ksp_type gmres -s_ksp_converged_reason -schurgmg lower" 1 6

runstokes_7:
	-@../ftestit.sh stokes.py "-stokeshelp" 1 7

test_gmshversion: rungmshversion_1

test_stokes: runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 runstokes_6 runstokes_7

test: test_gmshversion test_stokes

# etc

.PHONY: clean rungmshversion_1 runstokes_1 runstokes_2 runstokes_3 runstokes_4 runstokes_5 runstokes_6 runstokes_7 test_stokes test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.pvtu *.vtu *.m maketmp tmp difftmp
