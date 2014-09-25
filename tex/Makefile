all: structure.pdf

# TO BUILD PDF:
#     ice-bib.bib  needs to be a link to   pism/doc/ice-bib.bib

%.pdf: %.tex %.aux
	pdflatex $<

%.aux: %.tex
	pdflatex $<
	bibtex $*

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.toc *.dvi
