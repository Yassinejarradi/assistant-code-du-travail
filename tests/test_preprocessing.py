"""French tokenizer (BM25) and chunking rules."""

import chunking
from text_utils import tokenize


def test_stemming_merges_word_forms():
    # stemming happens before accent stripping, so all forms share one stem
    assert tokenize("salarié") == tokenize("salariée") == tokenize("salariés") == ["salari"]
    assert tokenize("licenciement")[0] == tokenize("licencié")[0]


def test_tokenize_drops_stopwords_and_list_markers():
    assert tokenize("Le congé de la salariée") == ["cong", "salari"]
    assert tokenize("1. le préavis") == tokenize("préavis")   # "1" (list marker) and "le" are dropped
    assert "39" in tokenize("article 39")             # but article numbers are kept


def test_short_article_is_not_split():
    assert chunking.split_article("Un seul paragraphe.", max_chars=1500) == ["Un seul paragraphe."]


def test_long_article_parts_repeat_the_lead_in():
    lead = "Sont punis d'une amende de 2.000 à 5.000 dirhams :"
    items = [f"{i}. infraction numéro {i} " + "x" * 80 for i in range(1, 30)]
    parts = chunking.split_article("\n".join([lead, *items]), max_chars=500)
    assert len(parts) > 1
    assert all(p.startswith(lead) for p in parts)
    assert all(len(p) <= 500 + len(lead) for p in parts)


def test_amendment_note_uses_official_text_when_available():
    amd = {"law": "Loi n° 1.23", "bulletin_officiel": "B.O. n° 1", "in_force": "2026-01-01",
           "summary": "résumé", "official_text": None}
    assert "Résumé, texte officiel non reproduit" in chunking.amendment_note(amd)
    assert "Nouvelle rédaction : texte officiel" in chunking.amendment_note({**amd, "official_text": "texte officiel"})


def test_amendment_is_attached_to_every_part(monkeypatch):
    amd = {"article_number": 1, "law": "Loi n° 1.23", "bulletin_officiel": "B.O. n° 1",
           "in_force": "2026-01-01", "summary": "résumé", "official_text": None}
    monkeypatch.setattr(chunking, "load_amendments", lambda: {1: [amd]})
    article = {"id": "a1", "article_number": 1, "livre": "Livre I", "titre": None, "chapitre": None,
               "section": None, "sous_section": None, "page": 1,
               "text": "\n".join(["Chapeau :"] + [f"{i}. " + "y" * 200 for i in range(20)])}
    parts = chunking.build_chunks([article], max_chars=600)
    assert len(parts) > 1
    assert all("B.O. n° 1" in c["text"] and c["amended_by"] == ["Loi n° 1.23"] for c in parts)


def test_amendment_dates_are_written_in_french():
    assert chunking.french_date("2026-07-16") == "16 juillet 2026"
    amd = {"law": "Loi n° 1.23", "bulletin_officiel": "B.O. n° 1", "in_force": "2026-07-16",
           "summary": "résumé", "official_text": None}
    assert "en vigueur depuis le 16 juillet 2026" in chunking.amendment_note(amd)
