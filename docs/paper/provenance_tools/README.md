# Sentence-provenance tooling for `experiment.tex`

`experiment.tex` is generated: the prose lives in `prov.py` (one entry per sentence, with the published sentence whose
structure it follows) and the layout, tables and Chinese translations live in `experiment_template.tex`.

    python build.py ../experiment.tex            # template + prov.py -> experiment.tex
    python verify.py ../experiment.tex --rest    # every source sentence found verbatim; no unregistered prose
    python official_check.py                     # optional: compare with the official conference PDFs (needs pymupdf)
    python gen_md_exp.py ../experiment_provenance.md

`verify.py` expects the LaTeX sources of the cited papers under `../src/<key>/x/` relative to this folder's parent
(arXiv e-print archives, unpacked; `nature_drone` is the article text saved as `src/nature_drone/text.txt`).
The arXiv identifiers are listed in `PAPERS` at the top of `prov.py`.
