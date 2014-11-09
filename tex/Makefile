all: book.pdf

texinputs := getstarted.tex structured.tex unstructured.tex multigrid.tex scaling.tex stokes.tex constrained.tex

tikzinputs := samplepoly.1.tikz squarefour.tikz bump.1.tikz bump.poly.tikz mesh.1.tikz

pdfinputs := c2poisson-conv.pdf

hinputs := $(addprefix ../c/, convenience.h)

cinputs := structuredlaplacian.c readmesh.c poissontools.c c1e.c c1matvec.c c2poisson.c c3convert.c c3poisson.c c4poisson.c c9obstacle.c

cstrip := $(addprefix cstrip/, $(cinputs))

book.pdf: book.aux
	pdflatex book
	pdflatex book

book.aux: book.tex book.bib $(texinputs) $(hinputs) $(cstrip) $(tikzinputs) $(pdfinputs)
	pdflatex book
	bibtex book.aux

# The purpose of the following nonsense is to put only the "meat" of codes
# into the book.  We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"
cstrip/%.c: ../c/%.c
	rm -rf $@.tmp
	sed -e 's/ierr = //' $< > $@.tmp
	sed -i 's/CHKERRQ(ierr);//' $@.tmp
	sed -i '/\/\/STRIP/d' $@.tmp
	sed -i '/PetscErrorCode ierr/d' $@.tmp
	mv $@.tmp $@

samplepoly.1.tikz: samplepoly.poly
	triangle -pqa4.0 samplepoly
	python tri2tikz.py --labelnodes --nodesize 2.0 --nodeoffset -0.3 --labelelements --scale 0.6 samplepoly.1 $@

squarefour.tikz: squarefour.poly
	python tri2tikz.py --labelnodes --nodesize 0.5 --nodeoffset -0.1 --scale 3.0 squarefour $@

bump.1.tikz: ../c/bump.poly
	(cd ../c/ && triangle -pqa1.0 bump)
	rm -rf bump.1.node bump.1.ele bump.1.poly
	ln -s ../c/bump.1.node
	ln -s ../c/bump.1.ele
	ln -s ../c/bump.1.poly
	python tri2tikz.py --labelnodes --nodeoffset 0.25 --labelelements --scale 1.7 bump.1 $@

bump.poly.tikz: ../c/bump.poly
	rm -rf bump.poly
	ln -s ../c/bump.poly
	python tri2tikz.py --scale 0.85 --polyonly bump --nodesize 3.0 bump.poly.tikz

mesh.1.tikz: ../c/genstructured.py
	(cd ../c/ && python genstructured.py mesh.1 5)
	rm -rf mesh.1.node mesh.1.ele mesh.1.poly
	ln -s ../c/mesh.1.node
	ln -s ../c/mesh.1.ele
	ln -s ../c/mesh.1.poly
	python tri2tikz.py --labelnodes --nodeoffset 0.05 --nodesize 0.5 --labelelements --scale 5.0 mesh.1 $@

c2poisson-conv.pdf: c2poisson-conv.py
	python c2poisson-conv.py

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	@rm -f cstrip/*.c
	@rm -f bump.?.* samplepoly.?.* bump.poly* squarefour.tikz mesh.?.*

