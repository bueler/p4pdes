all: book.pdf

texinputs := getstarted.tex linearsystem.tex structured.tex unstructured.tex multigrid.tex scaling.tex stokes.tex constrained.tex

tikzinputs := samplepoly.1.tikz squarefour.tikz bump.1.tikz bump.poly.tikz mesh.1.tikz

pdfinputs := poisson-conv.pdf richpolyplot.pdf

hinputs :=

# for these, see strip.sh
cinputs := structuredpoisson.c poisson.c readmesh.c poissontools.c poissonfem.c fish2.c obstacle.c
cstrip := $(addprefix cstrip/, $(cinputs))

book.pdf: book.aux
	pdflatex book
	pdflatex book

book.aux: book.tex book.bib $(texinputs) $(hinputs) $(cstrip) $(tikzinputs) $(pdfinputs)
	pdflatex book
	bibtex book.aux

$(cstrip):
	./strip.sh

samplepoly.1.tikz: samplepoly.poly
	triangle -pqa4.0 samplepoly
	python tri2tikz.py --labelnodes --nodesize 2.0 --nodeoffset -0.3 --labelelements --scale 0.6 samplepoly.1 $@

squarefour.tikz: squarefour.poly
	python tri2tikz.py --labelnodes --nodesize 0.5 --nodeoffset -0.1 --scale 3.0 squarefour $@

bump.1.tikz: ../c/ch4/bump.poly
	(cd ../c/ch4/ && triangle -pqa1.0 bump)
	python tri2tikz.py --labelnodes --nodeoffset 0.25 --labelelements --scale 1.7 ../c/ch4/bump.1 $@

bump.poly.tikz: ../c/ch4/bump.poly
	python tri2tikz.py --scale 0.85 --polyonly ../c/ch4/bump --nodesize 3.0 bump.poly.tikz

mesh.1.tikz: ../c/ch4/genstructured.py
	(cd ../c/ch4/ && python genstructured.py mesh.1 5)
	python tri2tikz.py --labelnodes --nodeoffset 0.05 --nodesize 0.5 --labelelements --scale 5.0 ../c/ch4/mesh.1 $@

poisson-conv.pdf: poisson-conv.py
	python poisson-conv.py

richpolyplot.pdf: richpolyplot.py
	python richpolyplot.py

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	@rm -rf cstrip/
	@rm -f bump.?.* samplepoly.?.* bump.poly* squarefour.tikz mesh.?.*
	@rm -f c2poisson-conv.pdf richpolyplot.pdf

