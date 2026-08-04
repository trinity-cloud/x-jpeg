SVG := $(wildcard figures/*.svg)
PDF_FIGURES := $(patsubst figures/%.svg,figures/pdf/%.pdf,$(SVG))

.PHONY: all figures clean

all: xjpeg-paper.pdf

figures: $(PDF_FIGURES)

figures/pdf/%.pdf: figures/%.svg
	@mkdir -p figures/pdf
	xmllint --noout $<
	rsvg-convert -f pdf -o $@ $<

build/main.pdf: main.tex references.bib $(PDF_FIGURES)
	@mkdir -p build
	tectonic -X compile main.tex --outdir build

xjpeg-paper.pdf: build/main.pdf
	cp build/main.pdf $@

clean:
	rm -rf build
	rm -f xjpeg-paper.pdf figures/pdf/*.pdf
