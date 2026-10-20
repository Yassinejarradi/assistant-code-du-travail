"""Extract the articles of the Moroccan Labour Code (Loi 65-99) from the official PDF.

Output: data/articles.json — one record per article with its place in the
hierarchy (livre > titre > chapitre > section > sous-section) and its footnotes.

The PDF is born-digital (Word 2016), so we rely on PyMuPDF's span-level font
information rather than on plain text:
    - body text ............ Book Antiqua 14pt
    - article headings ..... "Article N" (bold 15pt, but two are regular 14pt,
                             so headings are matched by text pattern)
    - livre ................ bold 18pt; titre/chapitre bold 16pt; section bold 14pt
    - footnote markers ..... superscript spans (flag bit 0)
    - footnotes / page nos . Times New Roman 12pt at the bottom of the page

Usage:
    python src/extract_articles.py [--pdf PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "data" / "raw" / "code_du_travail_2011.pdf"
DEFAULT_OUT = ROOT / "data" / "articles.json"

SOURCE = (
    "Loi n° 65-99 relative au Code du travail, promulguée par le dahir "
    "n° 1-03-194 du 11 septembre 2003 — version consolidée (adala.justice.gov.ma)"
)
VERSION_DATE = "2011-10-26"
LAST_ARTICLE = 589

LEVELS = ["livre", "titre", "chapitre", "section", "sous_section"]
HEADING_RE = re.compile(r"^(Livre|Titre|Chapitre|Sous-section|Section)\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Article\s+(premier|\d+)\s*$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^[-\s]*\d*[-\s]*$")
FOOTNOTE_START_RE = re.compile(r"^(\d{1,3})\s*[-–]?\s+(.*)$")
LIST_ITEM_RE = re.compile(r"^(\d+[.)°]|[a-z]\)|[-–•*])\s")
ROMAN_WITH_L_RE = re.compile(r"\b([IVX]*l[IVXl]*)\b")  # "Il" typed instead of "II"

# A vertical gap larger than this between two body lines starts a new paragraph
# (regular line spacing is ~17pt, paragraph spacing ~24pt).
PARAGRAPH_GAP = 21.0
# Lines below this y are in the page-footer zone (page numbers sit at y≈787).
FOOTER_Y = 770.0
# Non-superscript text smaller than this is footnote text (body is 14pt, list bullets 12pt).
FOOTNOTE_MAX_SIZE = 11.5
# A number glued to a word, e.g. "professionnels102;" — a footnote marker not set as superscript.
GLUED_MARKER_RE = re.compile(r"(?<=[^\W\d])(\d{1,3})(?=[\s;:,.)]|$)")
# "l'article279" -> "l'article 279" (missing space in the source).
ARTICLE_REF_SPACE_RE = re.compile(r"\b(articles?)(\d)", re.IGNORECASE)


@dataclass
class Line:
    text: str          # text without superscript footnote markers
    markers: list[int]  # footnote numbers referenced on this line
    bold: bool
    is_footnote_font: bool
    x0: float
    y0: float
    page: int


@dataclass
class Article:
    number: int
    hierarchy: dict[str, str | None]
    page_start: int
    lines: list[Line] = field(default_factory=list)
    note_refs: list[int] = field(default_factory=list)


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def normalize_heading(text: str) -> str:
    """'Chapitre Il: De la ...' -> 'Chapitre II : De la ...'."""
    text = clean(text)
    keyword, _, rest = text.partition(" ")
    number, sep, title = rest.partition(":")
    number = ROMAN_WITH_L_RE.sub(lambda m: m.group(1).replace("l", "I"), number.strip())
    if number.lower() == "premier":
        number = "premier"
    head = f"{keyword.capitalize()} {number}".strip()
    return f"{head} : {title.strip()}" if sep else head


def read_lines(doc: pymupdf.Document) -> list[Line]:
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for raw in block.get("lines", []):
                spans = [s for s in raw["spans"] if s["text"].strip()]
                if not spans:
                    continue
                y0 = raw["bbox"][1]
                # Footnotes are Times 12pt, except one (n°48) continued in Book Antiqua 10pt.
                main = max((s for s in spans if not s["flags"] & 1), key=lambda s: len(s["text"]), default=spans[0])
                is_footnote_font = "Times" in main["font"] or main["size"] < FOOTNOTE_MAX_SIZE
                full = "".join(s["text"] for s in spans)
                if y0 > FOOTER_Y and PAGE_NUMBER_RE.match(full):
                    continue  # page number "- 94 -"
                markers, parts = [], []
                for s in raw["spans"]:
                    is_superscript = bool(s["flags"] & 1) and not is_footnote_font
                    if is_superscript and s["text"].strip().isdigit():
                        markers.append(int(s["text"].strip()))
                    else:
                        parts.append(s["text"])
                text = clean("".join(parts))
                if not text:
                    continue
                bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans if not s["flags"] & 1)
                lines.append(Line(text, markers, bold, is_footnote_font, raw["bbox"][0], y0, page_index + 1))
    return lines


def collect_footnotes(lines: list[Line]) -> dict[int, str]:
    """Footnotes are numbered continuously through the document, so a line opens a
    new footnote only if it starts with the next number; otherwise it continues the
    previous one (wrapped lines can start with digits, e.g. "loi n° 17-" / "97 relative à")."""
    notes: dict[int, str] = {}
    current: int | None = None
    last = 0
    for line in lines:
        if not line.is_footnote_font:
            continue  # keep `current`: a long footnote may continue at the bottom of the next page
        m = FOOTNOTE_START_RE.match(line.text)
        if m and int(m.group(1)) == last + 1:
            current = last = int(m.group(1))
            notes[current] = m.group(2)
        elif current is not None:
            notes[current] += " " + line.text
        # else: Times text before footnote 1 (cover page) — not a footnote
    return {k: clean(v) for k, v in notes.items()}


def lines_to_text(lines: list[Line]) -> str:
    """Rebuild paragraphs: join wrapped lines, break on paragraph gaps and list items."""
    paragraphs: list[str] = []
    prev: Line | None = None
    for line in lines:
        text = line.text
        if prev is None:
            new_paragraph = True
        elif line.page != prev.page:
            # No reliable gap across a page break: use indentation and case instead.
            new_paragraph = bool(LIST_ITEM_RE.match(text)) or (line.x0 > 90 and not text[0].islower())
        else:
            new_paragraph = (line.y0 - prev.y0) > PARAGRAPH_GAP or bool(LIST_ITEM_RE.match(text))
        if new_paragraph or not paragraphs:
            paragraphs.append(text)
        elif paragraphs[-1].endswith("-") and not paragraphs[-1].endswith(" -"):
            paragraphs[-1] += text  # "saisie-" + "arrêt"
        else:
            paragraphs[-1] += " " + text
        prev = line
    return ARTICLE_REF_SPACE_RE.sub(r"\1 \2", "\n".join(paragraphs))


def parse(doc: pymupdf.Document) -> tuple[list[Article], dict[int, str]]:
    lines = read_lines(doc)
    footnotes = collect_footnotes(lines)
    body = [l for l in lines if not l.is_footnote_font]

    hierarchy: dict[str, str | None] = {lvl: None for lvl in LEVELS}
    articles: list[Article] = []
    current: Article | None = None  # article receiving body lines; None right after a heading
    last_number = 0
    pending_refs: list[int] = []  # markers seen on headings, before the next article
    started = False
    open_heading: str | None = None  # level whose heading may continue on the next bold line
    last_marker = 0  # footnote markers appear in increasing order through the document

    for line in body:
        # Markers are numbered in sequence, so a number glued to a word that equals the
        # next expected marker is a footnote marker that wasn't set as superscript.
        m = GLUED_MARKER_RE.search(line.text)
        if m and int(m.group(1)) == last_marker + 1 and int(m.group(1)) in footnotes:
            line.markers.append(int(m.group(1)))
            line.text = line.text[: m.start()] + line.text[m.end():]
        last_marker = max([last_marker, *line.markers])

        if not started:
            if line.bold and line.text.lower().startswith("livre"):
                started = True
            else:
                continue  # front matter: cover, dahir, préface, préambule

        m = ARTICLE_RE.match(line.text)
        expected = last_number + 1
        if m:
            number = 1 if m.group(1).lower() == "premier" else int(m.group(1))
            if number == expected:
                current = Article(number, dict(hierarchy), line.page)
                last_number = number
                current.note_refs.extend(pending_refs + line.markers)
                pending_refs = []
                articles.append(current)
                open_heading = None
                continue
            print(f"warning: p{line.page} '{line.text}' out of sequence (expected {expected}); kept as text",
                  file=sys.stderr)

        h = HEADING_RE.match(line.text)
        if line.bold and h:
            level = h.group(1).lower().replace("-", "_")
            hierarchy[level] = normalize_heading(line.text)
            for lower in LEVELS[LEVELS.index(level) + 1:]:
                hierarchy[lower] = None
            pending_refs.extend(line.markers)
            open_heading = level
            current = None
            continue
        if line.bold and open_heading:  # wrapped heading, e.g. "... de la" / "rémunération du salarié"
            hierarchy[open_heading] = clean(f"{hierarchy[open_heading]} {line.text}")
            pending_refs.extend(line.markers)
            continue
        open_heading = None

        if current is None:
            print(f"warning: p{line.page} text outside any article dropped: {line.text[:60]!r}", file=sys.stderr)
            continue
        current.lines.append(line)
        current.note_refs.extend(line.markers)

    return articles, footnotes


def to_records(articles: list[Article], footnotes: dict[int, str]) -> list[dict]:
    records = []
    for art in articles:
        text = lines_to_text(art.lines)
        notes = [{"number": n, "text": footnotes.get(n, "")} for n in dict.fromkeys(art.note_refs)]
        abrogated = not text and any("abrog" in n["text"].lower() for n in notes)
        records.append({
            "id": f"code-travail-art-{art.number}",
            "article_number": art.number,
            "livre": art.hierarchy["livre"],
            "titre": art.hierarchy["titre"],
            "chapitre": art.hierarchy["chapitre"],
            "section": art.hierarchy["section"],
            "sous_section": art.hierarchy["sous_section"],
            "text": "[Abrogé]" if abrogated else text,
            "abrogated": abrogated,
            "notes": notes,
            "page": art.page_start,
            "source": SOURCE,
            "version_date": VERSION_DATE,
        })
    return records


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with pymupdf.open(args.pdf) as doc:
        articles, footnotes = parse(doc)
    records = to_records(articles, footnotes)

    missing_notes = sorted({n["number"] for r in records for n in r["notes"] if not n["text"]})
    if missing_notes:
        print(f"warning: footnote markers without footnote text: {missing_notes}", file=sys.stderr)
    if records and records[-1]["article_number"] != LAST_ARTICLE:
        print(f"warning: last article is {records[-1]['article_number']}, expected {LAST_ARTICLE}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} articles and {len(footnotes)} footnotes parsed -> {args.out}")


if __name__ == "__main__":
    main()
