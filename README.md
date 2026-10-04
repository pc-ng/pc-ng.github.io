# Pai Chet Ng — academic website

A lightweight, static academic website published with GitHub Pages from the repository's `master` branch.

## Preview

```sh
python3 scripts/serve_preview.py
```

Open http://127.0.0.1:8765. No package installation is needed to build or preview.

## Update content

- `data/publications.json`: complete references, DOI/source metadata, PDF paths, resources, and BibTeX.
- `data/projects.json`: research summaries, figures, credits, and project resources.
- `data/notes.json`: preserved original blog articles.
- `scripts/build_site.py`: shared layout and other page content.
- `asset/site.css`: shared responsive styling.
- `asset/paper/`: locally available papers.
- `asset/images/`: research figure crops and the generated avatar.
- `asset/slides/`: available presentation slides.

After changing content, run `python3 scripts/build_site.py`. Generated pages require no JavaScript to display their content. JavaScript adds search/filtering, citation copying, and equation formatting in the two archival technical notes. KaTeX 0.19.0 and its fonts are self-hosted under `asset/vendor/katex/`, with the upstream MIT license retained.

The AI avatar is based on the SIT faculty portrait supplied by the owner and was generated with OpenAI’s built-in image-generation tool. Its visible credit identifies OpenAI image generation; the tool does not expose an exact model version. The generation prompt and provenance are saved outside the public site in `../pc-ng-webpage-review/ai-avatar-provenance.md` and `../pc-ng-webpage-review/ai-avatar-cartoon-provenance.md`. The current avatar revision is recorded in `../pc-ng-webpage-review/ai-avatar-v3-provenance.md`.

Research figure crops are attributed and link back to the original publication or project. Paper PDFs are verified as PDFs before being saved. Bibliographic metadata is sourced from the supplied CV, publisher records, and SIT's institutional repository; missing DOIs are not guessed.

## Preservation and review

The unchanged original site is preserved on `backup/2026-10-03-original` and in an archive outside this directory. Legacy publication and research URLs redirect to their new locations. Existing paper URLs remain available. The original CV PDF is preserved outside the publicly served site.

The working branch is `redesign/academic-2026`; approved snapshots are published to `master`. Owner review notes and lists of papers still to supply are in `../pc-ng-webpage-review/`, outside the public site.

Collection scripts use `requests` and `beautifulsoup4` only for optional bibliography updates. They are not required to build or preview. Network collection is explicit, not part of the website build.

The current avatar continuously animates one unchanged portrait with local SVG displacement fields for hair and a subtle smile. Rebuild it with `python3 scripts/build_smooth_avatar.py`. Reduced-motion preferences retain the still portrait. The previous frame sequence has been replaced, and the visible pause/play control has been removed at the owner's request. Source and animation notes are in `../pc-ng-webpage-review/ai-avatar-smooth-provenance.md`.

The Research page places funded projects immediately after the introduction. Related-paper venue labels are stored as `research_venue` in the publication data. The original Mobile & IoT figure crops come from J13 Figure 1 (page 2), J8 Figure 1 (page 4), and J14 Figure 8 (page 5). Agentic Workflow uses Figure 2 from the MAR-12 / Beyond a Joke paper.

All Publications includes every paper, grouped by year, with two references initially visible in each year. Available papers have local PDF download links; remaining manuscripts are marked PDF pending. Save supplied manuscripts in `asset/paper/` as their publication ID (for example, `c10.pdf`), then rebuild to make them available. Teaching groups both SIT courses and clearly labelled HKUST teaching-assistant roles into Undergraduate and Postgraduate sections.

Notes retain two substantive posts from the owner's `pc-ng/pc-blog` repository: Dynamic Time Warping and Heart Rate Measurement from ECG. Original illustrations are stored in `asset/notes/`. The contact-tracing note explains research contribution and public visibility using verified coverage links. The retired histogram article redirects to Notes. Source/link verification details are kept outside the website in `../pc-ng-webpage-review/`.
