include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


unfem: unfem.o um.o chkopts
	-${CLINKER} -o unfem unfem.o um.o  ${PETSC_LIB}
	${RM} unfem.o um.o

petscPyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py

meshes/trap1.vec meshes/trap1.is: meshes/trap.geo msh2petsc.py
	-@gmsh -2 meshes/trap.geo -o meshes/trap1.msh > /dev/null
	-@./msh2petsc.py meshes/trap1.msh > /dev/null

meshes/trapneu1.vec meshes/trapneu1.is: meshes/trapneu.geo msh2petsc.py
	-@gmsh -2 meshes/trapneu.geo -o meshes/trapneu1.msh > /dev/null
	-@./msh2petsc.py meshes/trapneu1.msh > /dev/null

# testing
rununfem_1: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_case 0" 1 1

rununfem_2: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_quaddegree 2 -un_case 1" 1 2

rununfem_3: petscPyScripts meshes/trapneu1.vec meshes/trapneu1.is
	-@../testit.sh unfem "-un_mesh meshes/trapneu1 -un_case 2" 1 3

# FIXME need to rewrite genstructured.py
rununfem_4: petscPyScripts
	-@./genstructured.py meshes/square.1 3 > /dev/null
	-@./tri2petsc.py meshes/square.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/square.1 -un_case 3" 1 4

rununfem_5: petscPyScripts meshes/trap1.vec meshes/trap1.is
	-@../testit.sh unfem "-un_mesh meshes/trap1 -un_view_mesh" 1 5

test_unfem: rununfem_1 rununfem_2 rununfem_3 rununfem_4 rununfem_5

test: test_unfem

# etc

.PHONY: distclean rununfem_1 rununfem_2 rununfem_3 rununfem_4 rununfem_5 test test_unfem petscPyScripts

distclean:
	@rm -f *~ unfem *tmp
	@rm -f *.pyc PetscBinaryIO.py petsc_conf.py
	(cd meshes/ && ${MAKE} clean)
	(cd koch/ && ${MAKE} clean)

