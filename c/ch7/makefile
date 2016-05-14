include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


unfem: unfem.o um.o chkopts
	-${CLINKER} -o unfem unfem.o um.o  ${PETSC_LIB}
	${RM} unfem.o um.o

# testing
rununfem_1:
	-@triangle -pqa0.5 meshes/trap > /dev/null
	-@./tri2petsc.py meshes/trap.1 meshes/trap.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/trap.1 -un_quaddeg 1 -un_case 0" 1 1

rununfem_2:
	-@triangle -pqa0.5 meshes/trap > /dev/null
	-@./tri2petsc.py meshes/trap.1 meshes/trap.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/trap.1 -un_quaddeg 2 -un_case 1" 1 2

rununfem_3:
	-@triangle -pqa0.5 meshes/trapneu > /dev/null
	-@./tri2petsc.py meshes/trapneu.1 meshes/trapneu.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/trapneu.1 -un_quaddeg 1 -un_case 2" 1 3

rununfem_4:
	-@./genstructured.py meshes/square.1 3 > /dev/null
	-@./tri2petsc.py meshes/square.1 meshes/square.1 > /dev/null
	-@../testit.sh unfem "-un_mesh meshes/square.1 -un_quaddeg 1 -un_case 3" 1 4

test: rununfem_1 rununfem_2 rununfem_3 rununfem_4

# etc

.PHONY: distclean rununfem_1 rununfem_2 rununfem_3 rununfem_4 test

distclean:
	@rm -f *~ *.pyc unfem
	(cd meshes/ && ${MAKE} clean)

