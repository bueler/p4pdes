all: book.pdf

texroots := preface getstarted linearsystem structured unstructured multigrid scaling nonlinear stokes dmplex constrained
texinputs := $(addsuffix .tex, $(texroots))

tikzinputs := samplepoly.1.tikz squarefour.tikz bump.1.tikz bump.poly.tikz mesh.1.tikz

pdfinputs := poisson-conv.pdf poisson-cg-scale.pdf richpolyplot.pdf

hinputs :=

# for these, see strip.sh
cinputs := structuredpoisson.c poisson.c reaction.c fish2.c readmesh.c poissontools.c poissonfem.c obstacle.c
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

bump.1.tikz: ../c/ch7/bump.poly
	(cd ../c/ch7/ && triangle -pqa1.0 bump)
	python tri2tikz.py --labelnodes --nodeoffset 0.25 --labelelements --scale 1.7 ../c/ch7/bump.1 $@

bump.poly.tikz: ../c/ch7/bump.poly
	python tri2tikz.py --scale 0.85 --polyonly ../c/ch7/bump --nodesize 3.0 bump.poly.tikz

mesh.1.tikz: ../c/ch7/genstructured.py
	(cd ../c/ch7/ && python genstructured.py mesh.1 5)
	python tri2tikz.py --labelnodes --nodeoffset 0.05 --nodesize 0.5 --labelelements --scale 5.0 ../c/ch7/mesh.1 $@

richpolyplot.pdf: richpolyplot.py
	python richpolyplot.py

poisson-conv.pdf: poisson-conv.py
	python poisson-conv.py

poisson-cg-scale.pdf: poisson-cg-scale.py
	python poisson-cg-scale.py

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	@rm -rf cstrip/
	@rm -f bump.?.* samplepoly.?.* bump.poly* squarefour.tikz mesh.?.*
	@rm -f poisson-conv.pdf poisson-cg-scale.pdf richpolyplot.pdf

