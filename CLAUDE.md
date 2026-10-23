# CLAUDE.md

## Project

RAG chatbot answering questions about the **Moroccan Labour Code (Loi n° 65-99, Code du travail)**,
built for an AI master's project. Its core requirement is that **every answer cites the exact
article(s)** it is based on (e.g. "art. 152"), so retrieval works at the article level and each
chunk keeps its article number and hierarchy.

Source of truth: `data/raw/code_du_travail_2011.pdf`, the official French consolidated version
of 26 Oct 2011 from adala.justice.gov.ma. It has 589 articles in 7 livres plus the Livre préliminaire.

## Current phase

- Step 1 — extraction: done.
- Step 2 — retrieval: done (see `docs/02_retrieval.md`).
- Step 3 — answer generation: done (`src/rag.py`, local Ollama `qwen3:4b-instruct`).
- Step 4 — web UI: done (`app.py`, `streamlit run app.py`; bound to localhost in `.streamlit/config.toml`).

**Scope decision (2026-10-06): French Code du travail only.** `data/raw/code_travail arabic.pdf`
(Arabic code, text-based but with broken لا ligatures and RTL issues; version vs 2011 FR not yet
checked) and `data/raw/loi de grève 97.15.pdf` (strike law, scanned → needs Arabic OCR) are
parked for later. Don't process or index them until the user asks. Planned order when resumed:
Arabic questions on the FR index (cross-lingual baseline) → parallel `text_ar` per article →
strike law with its own `source`.

The user works step by step and wants to **learn**: explain concepts and trade-offs while
building, back decisions with measurements, and don't jump ahead to the next step unasked.

## Commands

```bash
.venv/Scripts/python src/extract_articles.py    # PDF -> data/articles.json
.venv/Scripts/python src/validate_articles.py   # must end with "✓ OK" (exit 0)
```

Always rerun the validator after changing the extractor.

```bash
.venv/Scripts/python src/chunking.py              # articles -> data/chunks.json
.venv/Scripts/python src/build_index.py           # bge-m3 -> Chroma (~10 min CPU); rerun after chunking changes
.venv/Scripts/python src/retriever.py "question"  # top-k articles
.venv/Scripts/python src/evaluate_retrieval.py    # Recall@k / MRR; rerun after any retrieval change
.venv/Scripts/python src/rag.py "question"        # full answer (needs Ollama running)
.venv/Scripts/python -m pytest                   # 42 tests, ~1 s, no models; also run by GitHub Actions
```

Run the tests after any change to the data, chunking, citation parsing or router rules.
Authors: **Yassine Jarradi** and **Soukaina Nejmaoui**, listed as plain names (no "en collaboration" wording).

## Generation (step 3) and local setup

- LLM: Ollama `qwen3:4b-instruct`, temperature 0.1, top-5 articles in the prompt. Don't use the
  plain `qwen3:4b` tag: it's a thinking model that ignores `think=False` and takes minutes.
- The system prompt (in `rag.py`, French) says: answer only from the articles, cite (Article N),
  say so when the articles don't answer. A post-check flags cited articles that are neither
  retrieved nor referenced inside a retrieved article.
- Machine: i7-8750H, 16 GB RAM, GTX 1050 Ti 4 GB, C: almost full. Caches live on D:
  (`HF_HOME=D:\ai-cache\huggingface`, `OLLAMA_MODELS=D:\ai-cache\ollama`, both user env vars).
  If Ollama is not running: `ollama serve`. Slowness is usually low free RAM (Chrome/games open).
- GPU: Ollama ≥ 0.40 defaults to CUDA 13, which dropped the GTX 1050 Ti (compute 6.1) and silently
  runs on CPU (nvidia-smi shows 0 MiB; prompt reading ~54 tok/s). `OLLAMA_LLM_LIBRARY=cuda_v12`
  (user env var + set in `lancer_assistant.bat`) restores the GPU: 27/37 layers, ~680 tok/s prompt.
  Check with `nvidia-smi` (~2.3 GB used while the model is loaded).
- Speed: `RAG.warm_up()` runs when the Streamlit page opens (~55 s once), and `KEEP_ALIVE="30m"`
  stops Ollama unloading the LLM after 5 min idle. Warm questions take ~5–15 s.
- Prompt lessons (4B model): the "no conversion/calculation" rule fixed invented unit conversions.
  An "answer about the person asking" rule made it answer in the first person, so it was removed.
  Known miss: "ma femme accouche" as a FIRST question → answers art. 152 although art. 269 is
  retrieved first (as a follow-up, rewriting fixes it).
- Keep it simple: this is a module/CV project, so use good defaults over option menus and
  extra benchmarks.

## Retrieval design (step 2)

- Chunks: 1 article = 1 chunk; articles > 1,500 chars are split at paragraph boundaries,
  repeating the first paragraph. `embed_text` = hierarchy path + "Article N" + text.
- Default mode `dense`: bge-m3 + Chroma (cosine), plus regex lookup for "article N".
  BM25 is available in `hybrid` mode, but every BM25 weight > 0 lowered recall on the eval set,
  so it stays at weight 0.1.
- Baseline to beat (dense): R@1 0.74, R@5 0.92, MRR 0.81 on `data/eval/questions.json`.
  That set was drafted by Claude and is not yet reviewed by the user: don't tune on it heavily.

## Amendments (post-2011)

- `data/amendments.json` holds changes published after the 2011 PDF. `chunking.py` appends each as
  a labelled note to EVERY chunk of its article (so no chunk shows the old rule alone).
  `articles.json` is never hand-edited.
- Current: loi n° 32.26 (Dahir 1.26.27 of 3 July 2026, B.O. n° 7526 of 16 July 2026, p. 4422) →
  art. 193: private security agents (loi 27-06 companies) work 8 h/day, not 12; 9-month
  transition for existing contracts. Only a press-based summary is stored (`official_text` is null):
  never invent the official wording.
- After editing: `chunking.py` → `build_index.py --update` (incremental, seconds) → restart the app.
- The app reloads data when `data/chunks.json` or `data/index/BUILD_ID` change (`build_index.py`
  rewrites BUILD_ID only when the index really changes). Never key the reload on Chroma's own
  files: their mtimes change on every read-only query, which made the app reload after each question.

## Web UI notes

- ChatGPT/Claude-style layout: centred greeting when empty (no example buttons on the main page,
  user request), user bubbles right-aligned via CSS `:has([data-testid=stChatMessageAvatarUser])`,
  sidebar = logo, new chat, session history, examples/about in expanders, author at the bottom.
- `st.chat_input` is read at the TOP of `app.py` so the greeting and sidebar title are correct
  in the same run (no `st.rerun`). Streamlit's promo popup is hidden (`stSkillsNudgeAnchor`).
- `fileWatcherType = "none"` in config: restart the server after editing `app.py`.
- Follow-ups: `RAG.condense()` rewrites the question with the last exchange before retrieval
  (« Et pour le père ? » → « congé de paternité »); the app shows « 🔎 Recherche : … » when rewritten.
- Router: `RAG.is_legal()` — regex for obvious small talk (instant), legal-keyword regex, else a
  1-word LLM classification (default legal when unclear). Small talk → `RAG.chat()` with few-shot
  examples (a 4B model ignored style rules: used "tu", emojis, answered off-topic). No search, no
  articles shown. Citations/sources are displayed only when the answer cites an article.
- Follow-up rewriting uses only legal exchanges as history ("Bonjour" made it rewrite full questions).
- Citation parsing: singular "Article N" = one article ("Article 231, 1" is point 1, not art. 1);
  only plural "Articles" introduces lists.
- Clickable citations: one `st.popover` per cited article, full text from `articles.json` +
  amendment notes (`articles_by_number`), so cross-referenced articles work too.
- README visuals (`docs/images/home.png`, `chat.png`, `demo.gif`) are recorded from the running app
  with Playwright (driving the installed Chrome).

## Data contract — `data/articles.json`

One record per article. Fields: `id`, `article_number`, `livre`, `titre`, `chapitre`, `section`,
`sous_section`, `text`, `abrogated`, `notes` (footnotes `[{number, text}]`), `page`,
`source`, `version_date`. Later steps build on this schema, so keep changes backward compatible.

## Facts about the PDF that the extractor relies on

- Text-based (Word 2016), so no OCR. PyMuPDF is used for its span-level font, size and superscript flags.
- Article headings: `Article N` / `Article premier`. They are usually bold 15pt, but art. 156 and 458
  are regular text, so headings are detected by regex plus sequence continuity, not by font alone.
- Footnote markers are superscript spans that are glued to words in plain text (`Article 25635` is
  Article 256 + note 35). One marker (n°102, art. 586) is not superscript and is caught by the
  "next marker number" rule.
- Footnotes: Times 12pt at the bottom of the page. Note 48 continues in Book Antiqua 10pt. Footnote
  numbering runs 1–102 through the whole document.
- Only **art. 256 is repealed** (by loi 48-06). Art. 327 and 334 only look repealed in raw text
  because of glued markers.
- The source writes some Roman numerals with a lowercase L (`Chapitre Il`), which is normalised to `II`.

## Conventions

- Python 3.13, virtualenv in `.venv/`, dependencies in `requirements.txt`.
- Code and comments in English; legal content stays in French exactly as in the source.
  Don't "correct" the law's wording, only fix extraction artefacts.
- Scripts live in `src/`, are runnable as `python src/<script>.py`, and have `--help`.
- The project sits in a OneDrive folder, which can lock files: avoid `pip install --upgrade pip`
  inside the venv because it has failed with WinError 5/32.
