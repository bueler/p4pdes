all: book.pdf

# chapters of the book
chapsroots := preface getstarted linearsystem structured nonlinear optifem timesys advdif unstructured precond scaling stokes constrained dmplex
chapsinputs := $(addprefix chaps/, $(addsuffix .tex, $(chapsroots)))

# for these, see strip.sh
croots := vecmatksp tri poisson expcircle expcircleJAC reaction plap ode heat fish2 readmesh poissontools poissonfem obstacle
cstrip := $(addprefix cstrip/, $(addsuffix .c, $(croots)))

tmptikzroots := blob blob.1.elenum blob.1.nodenum squarefour mesh.1
tmptikzinputs := $(addprefix tmp/, $(addsuffix .tikz, $(tmptikzroots)))

pdfroots := petscghostvalues
pdfinputs := $(addprefix figs/, $(addsuffix .pdf, $(pdfroots)))

pngroots := dmM5N7 line-graph-tri matM5N7
pnginputs := $(addprefix figs/, $(addsuffix .png, $(pngroots)))

book.pdf: book.aux
	pdflatex book
	pdflatex book

book.aux: book.tex book.bib macros.tex tmp/ $(chapsinputs) $(cstrip) $(tmptikzinputs) $(pdfinputs) $(pnginputs)
	(cd figs/ && make)
	pdflatex book
	bibtex book.aux

tmp/:
	mkdir tmp/

$(cstrip):
	python strip.py

# location of naive unstructured mesh stuff:
UNLOC := ch8/

tmp/blob.tikz tmp/blob.1.elenum.tikz tmp/blob.1.nodenum.tikz: tmp/ ../c/$(UNLOC)blob.poly
	python tri2tikz.py --polyonly --nodesize 1.0 --scale 0.6 ../c/$(UNLOC)blob tmp/blob.tikz
	triangle -pqa4.0 ../c/$(UNLOC)blob
	python tri2tikz.py --labelelements --nodesize 1.0 --scale 0.6 ../c/$(UNLOC)blob.1 tmp/blob.1.elenum.tikz
	python tri2tikz.py --labelnodes --nodesize 3.0 --nodeoffset -0.5 --scale 0.6 ../c/$(UNLOC)blob.1 tmp/blob.1.nodenum.tikz

tmp/squarefour.tikz: tmp/ ../c/$(UNLOC)squarefour.poly
	python tri2tikz.py --labelnodes --nodesize 0.5 --nodeoffset -0.1 --scale 3.0 ../c/$(UNLOC)squarefour $@

tmp/mesh.1.tikz: tmp/ ../c/$(UNLOC)genstructured.py
	(cd ../c/$(UNLOC) && python genstructured.py mesh.1 5)
	python tri2tikz.py --labelnodes --nodeoffset 0.05 --nodesize 0.5 --labelelements --scale 5.0 ../c/$(UNLOC)mesh.1 $@

.PHONY: clean

clean:
	rm -f *~ .gitignore~
	rm -f tikz/*~ timing/*~ timing/poisson/*~ timing/tri/*~
	rm -f *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	rm -rf cstrip/ tmp/
	(cd chaps/ && make clean)
	(cd figs/ && make clean)

