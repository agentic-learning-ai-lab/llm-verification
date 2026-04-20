# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Static project page for the paper **"When Does Verification Pay Off? A Closer Look at LLMs as Solution Verifiers"** (Lu, Teehan, Jin, Ren — NYU; arXiv:2512.02304). Deployed at https://agenticlearning.ai/llm-verification/.

The entire site is a single `index.html` plus vendored CSS/JS assets under `static/`. There is **no build system, no package manager, no tests, and no backend**. Edits to `index.html` and `static/css/index.css` are the usual workflow.

## Branch layout

- `main` — near-empty branch (README + LICENSE only). Do not put site content here.
- `website` — the actual site source; this is what gets published. Expect to work on this branch.

When opening PRs, confirm the target: content changes almost always go to `website`, not `main`.

## Preview locally

```
python3 -m http.server 8000
# then open http://localhost:8000/
```

No build step — just reload the browser after edits.

## Code structure

- `index.html` — the whole page. Sections are plain `<section class="hero">` blocks (abstract, overview, each research question, BibTeX, footer). MathJax is loaded from CDN; write LaTeX with `\( ... \)` for inline and `\[ ... \]` for display.
- `static/css/index.css` — the only hand-authored stylesheet; everything else (`bulma.min.css`, `bulma-carousel.min.css`, `bulma-slider.min.css`, `fontawesome.all.min.css`) is vendored from the Academic Project Page Template and should not be edited.
- `static/js/index.js` — small init for Bulma carousel/slider. The rest of `static/js/` is vendored.
- `static/images/` — result figures embedded via `<embed src="static/images/...-1.png">`. Figure filenames follow the `results_<topic>-1.png` convention from the paper's plotting pipeline; keep that naming when adding new ones.
- `static/pdfs/`, `static/videos/` — placeholder dirs from the template; currently only a sample PDF.

## Template origin

Forked from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (Nerfies-derived, Bulma-based). Unexplained CSS classes or markup idioms usually trace back there — check before refactoring "dead" code.

## Conventions worth preserving

- Model-family color spans (`<span class="model-llama3">`, `model-qwen25`, `model-qwen3`, `model-deepseek`) are defined in `index.css` and used throughout the prose. Reuse them rather than inline-styling when mentioning families.
- Takeaway callouts use the gray-background / blue-left-border pattern (`background-color: #f5f5f5; border-left: 4px solid #3273dc`). Match this when adding new takeaways.
- Figures are centered with `<p style="text-align: center;"><embed ... width="..." height=auto></p>`. Widths in the current page range 500–900px depending on aspect ratio.
- Keep the GitHub link pointing at `https://github.com/agentic-learning-ai-lab/llm-verification` (the published repo), not this local clone's remote.
