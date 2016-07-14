all: book.pdf

# chapters of the book
chapsroots := preface getstarted linearsystem structured nonlinear optifem timesys advdif unstructured precond scaling stokes constrained dmplex
chapsinputs := $(addprefix chaps/, $(addsuffix .tex, $(chapsroots)))

# for these, see strip.sh
croots := vecmatksp tri poisson expcircle expcircleJAC reaction plap ode heat fish2 readmesh poissontools poissonfem obstacle
cstrip := $(addprefix cstrip/, $(addsuffix .c, $(croots)))

tmptikzroots := blob blob.1.elenum blob.1.nodenum trap.1 trap.2 mesh.1 koch koch.1
tmptikzinputs := $(addprefix tmp/, $(addsuffix .tikz, $(tmptikzroots)))

pdfroots := petscghostvalues
pdfinputs := $(addprefix figs/, $(addsuffix .pdf, $(pdfroots)))

pngroots := dmM5N7 line-graph-tri matM5N7 trap1mat trap2mat trap3mat
pnginputs := $(addprefix figs/, $(addsuffix .png, $(pngroots)))

book.pdf: book.aux
	pdflatex book
	pdflatex book

book.aux: book.tex book.bib macros.tex $(chapsinputs) $(cstrip) $(tmptikzinputs) $(pdfinputs) $(pnginputs)
	(cd figs/ && make)
	pdflatex book
	bibtex book.aux

tmp/:
	mkdir tmp/

$(cstrip):
	python strip.py

# location of naive unstructured mesh stuff:
UNLOC := ../c/ch7/

tmp/blob.tikz tmp/blob.1.elenum.tikz tmp/blob.1.nodenum.tikz: tmp/ $(UNLOC)meshes/blob.poly
	ln -sf $(UNLOC)tri2petsc.py
	python tri2tikz.py --polyonly --nodesize 1.0 --scale 0.6 $(UNLOC)meshes/blob tmp/blob.tikz
	triangle -pqa4.0 $(UNLOC)meshes/blob
	python tri2tikz.py --labelelements --nodesize 1.0 --scale 0.5 $(UNLOC)meshes/blob.1 tmp/blob.1.elenum.tikz
	python tri2tikz.py --labelnodes --nodesize 3.0 --nodeoffset -0.5 --scale 0.48 $(UNLOC)meshes/blob.1 tmp/blob.1.nodenum.tikz
	rm -f tri2petsc.py*

tmp/trap.1.tikz tmp/trap.2.tikz: tmp/ $(UNLOC)meshes/trap.poly
	ln -sf $(UNLOC)tri2petsc.py
	triangle -pqa0.5 $(UNLOC)meshes/trap
	python tri2tikz.py --nodesize 0.9 --scale 2.5 $(UNLOC)meshes/trap.1 tmp/trap.1.tikz
	triangle -rpqa0.1 $(UNLOC)meshes/trap.1
	python tri2tikz.py --nodesize 0.8 --scale 2.5 $(UNLOC)meshes/trap.2 tmp/trap.2.tikz
	rm -f tri2petsc.py*

tmp/mesh.1.tikz: tmp/ $(UNLOC)genstructured.py
	python $(UNLOC)genstructured.py mesh.1 4
	ln -sf $(UNLOC)tri2petsc.py
	python tri2tikz.py --labelnodes --nodeoffset 0.07 --nodesize 0.5 --scale 4.0 mesh.1 $@
	rm -f mesh.1.* tri2petsc.py*

tmp/koch.tikz tmp/koch.1.tikz: tmp/ $(UNLOC)koch/polygon.py
	ln -sf $(UNLOC)tri2petsc.py
	$(UNLOC)koch/polygon.py -l 2 -o $(UNLOC)koch/koch.poly
	python tri2tikz.py --polyonly --polydirichletwidth 1.0 --scale 3.0 $(UNLOC)koch/koch tmp/koch.tikz
	triangle -pqa0.025 $(UNLOC)koch/koch
	python tri2tikz.py --nodesize 0.2 --polydirichletwidth 1.0 --scale 3.0 $(UNLOC)koch/koch.1 tmp/koch.1.tikz
	rm -f tri2petsc.py*

.PHONY: clean

clean:
	rm -f *~ *.pyc .gitignore~
	rm -f tikz/*~ timing/*~ timing/poisson/*~ timing/tri/*~
	rm -f *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	rm -rf cstrip/ tmp/
	(cd chaps/ && make clean)
	(cd figs/ && make clean)

