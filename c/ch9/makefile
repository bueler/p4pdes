include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


convert: convert.o readmesh.o  chkopts
	-${CLINKER} -o convert convert.o readmesh.o  ${PETSC_KSP_LIB}
	${RM} convert.o readmesh.o

poissonfem: poissonfem.o readmesh.o poissontools.o chkopts
	-${CLINKER} -o poissonfem poissonfem.o readmesh.o poissontools.o ${PETSC_KSP_LIB}
	${RM} poissonfem.o readmesh.o poissontools.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ convert poissonfem
	@rm -f blob.?.* squarefour.?.* square.?.* mesh.?.*
