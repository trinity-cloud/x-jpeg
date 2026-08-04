# X-JPEG paper and figure package

This directory contains the technical paper for the X-JPEG 0.1.0 release and
its editable vector artwork.

## Primary artifacts

- `paper.md` — complete Markdown paper and web-friendly SVG embeds
- `main.tex` — two-column machine-learning preprint source
- `xjpeg-paper.pdf` — rendered paper
- `references.bib` — reusable bibliography
- `data/benchmark-summary.csv` — values used by the result tables/graphic
- `figures/*.svg` — editable source artwork
- `figures/png/*.png` — high-resolution 3,840-pixel-wide exports
- `figures/pdf/*.pdf` — vector conversions used by LaTeX

The layout is deliberately conference-neutral: it uses the compact two-column
visual language common to machine-learning preprints without claiming
submission to, or acceptance by, any venue.

## Build

Requirements:

- `rsvg-convert` (librsvg)
- `tectonic`

Run:

```bash
make -C paper clean all
```

The build validates each SVG as XML, converts it to vector PDF, compiles the
LaTeX source, and writes `paper/xjpeg-paper.pdf`.

## Evidence policy

The paper reports only results linked to the released 2026 model. The central
adaptivity result uses 1,000 held-out images and paired bootstrap confidence
intervals. The 0.5/0.25-bpp comparison is explicitly marked as a 100-image
exploratory pilot without paired intervals.
