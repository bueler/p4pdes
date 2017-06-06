include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


unfem: unfem.o um.o chkopts
	-${CLINKER} -o unfem unfem.o um.o  ${PETSC_LIB}
	${RM} unfem.o um.o

petscPyScripts:
	ln -sf ${PETSC_DIR}/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/bin/petsc_conf.py

# testing
rununfem_1: petscPyScripts
	-@triangle -pqa0.5 meshes/trap > /dev/null
	-@./tri2petsc.py meshes/trap.1 meshes/trap.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/trap.1 -un_case 0" 1 1

rununfem_2: petscPyScripts
	-@triangle -pqa0.5 meshes/trap > /dev/null
	-@./tri2petsc.py meshes/trap.1 meshes/trap.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/trap.1 -un_quaddegree 2 -un_case 1" 1 2

rununfem_3: petscPyScripts
	-@triangle -pqa0.5 meshes/trapneu > /dev/null
	-@./tri2petsc.py meshes/trapneu.1 meshes/trapneu.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/trapneu.1 -un_case 2" 1 3

rununfem_4: petscPyScripts
	-@./genstructured.py meshes/square.1 3 > /dev/null
	-@./tri2petsc.py meshes/square.1 meshes/square.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/square.1 -un_case 3" 1 4

test_unfem: rununfem_1 rununfem_2 rununfem_3 rununfem_4

test: test_unfem

# etc

.PHONY: distclean rununfem_1 rununfem_2 rununfem_3 rununfem_4 test test_unfem petscPyScripts

distclean:
	@rm -f *~ unfem *tmp
	@rm -f *.pyc PetscBinaryIO.py petsc_conf.py
	(cd meshes/ && ${MAKE} clean)
	(cd koch/ && ${MAKE} clean)

