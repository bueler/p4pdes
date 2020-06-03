include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

unfem: unfem.o um.o
	-${CLINKER} -o unfem unfem.o um.o  ${PETSC_LIB}
	${RM} unfem.o um.o

petscPyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py

meshes/trap1.vec meshes/trap1.is: meshes/trap.geo msh2petsc.py
	-@gmsh -2 meshes/trap.geo -o meshes/trap1.msh > /dev/null
	-@./msh2petsc.py meshes/trap1.msh > /dev/null

meshes/trap2.vec meshes/trap2.is: meshes/trap.geo msh2petsc.py
	-@gmsh -2 meshes/trap.geo -o meshes/trap2.msh > /dev/null
	-@gmsh -refine meshes/trap2.msh > /dev/null
	-@./msh2petsc.py meshes/trap2.msh > /dev/null

meshes/trapneu1.vec meshes/trapneu1.is: meshes/trapneu.geo msh2petsc.py
	-@gmsh -2 meshes/trapneu.geo -o meshes/trapneu1.msh > /dev/null
	-@./msh2petsc.py meshes/trapneu1.msh > /dev/null

meshes/square1.vec meshes/square1.is: genstructured.py
	-@./genstructured.py meshes/square1 3 > /dev/null

# testing
rununfem_1: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_case 0" 1 1

rununfem_2: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_quaddegree 2 -un_case 1 -un_view_mesh" 1 2

rununfem_3: petscPyScripts meshes/trapneu1.vec meshes/trapneu1.is
	-@../testit.sh unfem "-un_mesh meshes/trapneu1 -un_case 2" 1 3

rununfem_4: petscPyScripts meshes/square1.vec meshes/square1.is
	-@../testit.sh unfem "-un_mesh meshes/square1 -un_case 3" 1 4

rununfem_5: petscPyScripts meshes/trap2.vec meshes/trap2.is
	-@../testit.sh unfem "-un_mesh meshes/trap2 -pc_type gamg -ksp_converged_reason -ksp_rtol 1.0e-9" 1 5

rununfem_6: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_case 1 -snes_mf_operator -snes_converged_reason -ksp_converged_reason" 1 6

rununfem_7: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_case 1 -snes_fd_color -snes_converged_reason" 1 7

test_unfem: rununfem_1 rununfem_2 rununfem_3 rununfem_4 rununfem_5 rununfem_6 rununfem_7

test: test_unfem

# etc
.PHONY: distclean rununfem_1 rununfem_2 rununfem_3 rununfem_4 rununfem_5 rununfem_6 rununfem_7 test test_unfem petscPyScripts

distclean:
	@rm -f *~ unfem *tmp
	@rm -f *.pyc PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/
	(cd meshes/ && ${MAKE} clean)
	(cd koch/ && ${MAKE} clean)
	(cd vis/ && ${MAKE} clean)

