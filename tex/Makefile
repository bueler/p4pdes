all: book.pdf

tikzinputs := samplepoly.1.tikz squarefour.tikz bump.1.tikz bump.2.tikz
texinputs := getstarted.tex linear.tex

hinputs := $(addprefix ../c/, convenience.h)
cinputs := readmesh.c poissontools.c c1matvec.c c2triangle.c c2testprealloc.c
cstrip := $(addprefix cstrip/, $(cinputs))

# TO BUILD PDF:
#     ice-bib.bib  needs to be a link to   pism/doc/ice-bib.bib

book.pdf: book.tex book.aux $(texinputs) $(hinputs) $(cstrip) $(tikzinputs)
	pdflatex book
	pdflatex book

book.aux: book.tex $(texinputs) $(hinputs) $(cstrip) $(tikzinputs)
	pdflatex book
	bibtex book.aux

# The purpose of the following nonsense is to put only the "meat" of codes
# into the book.  We remove ierr stuff.
cstrip/%.c: ../c/%.c
	sed -e 's/ierr = //' $< > $@.tmp
	sed -e 's/ CHKERRQ(ierr);//' $@.tmp > $@
	rm -rf $@.tmp

samplepoly.1.tikz: samplepoly.poly tri2tikz.py
	triangle -pqa4.0 samplepoly
	python tri2tikz.py --labelnodes --nodesize 2.0 --nodeoffset -0.3 --labelelements --scale 0.6 samplepoly.1 $@

squarefour.tikz: squarefour.poly tri2tikz.py
	python tri2tikz.py --labelnodes --nodesize 0.5 --nodeoffset -0.1 --scale 3.0 squarefour $@

bump.1.tikz: ../c/bump.poly tri2tikz.py
	(cd ../c/ && triangle -pqa1.0 bump)
	rm -rf bump.1.node bump.1.ele bump.1.poly
	ln -s ../c/bump.1.node
	ln -s ../c/bump.1.ele
	ln -s ../c/bump.1.poly
	python tri2tikz.py --labelnodes --nodeoffset 0.25 --scale 2.0 bump.1 $@

bump.2.tikz: ../c/bump.poly ../c/bump.1.poly tri2tikz.py
	(cd ../c/ && triangle -rpqa0.25 bump.1)
	rm -rf bump.2.node bump.2.ele bump.2.poly
	ln -s ../c/bump.2.node
	ln -s ../c/bump.2.ele
	ln -s ../c/bump.2.poly
	python tri2tikz.py --scale 2.0 bump.2 $@

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi cstrip/*.c bump.?.* samplepoly.?.*
