"""Integrity of the generated data: articles, chunks, amendments and the evaluation set."""

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
LAST_ARTICLE = 589


@pytest.fixture(scope="module")
def articles():
    return json.loads((DATA / "articles.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    return json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))


def test_articles_are_continuous(articles):
    assert [a["article_number"] for a in articles] == list(range(1, LAST_ARTICLE + 1))


def test_only_article_256_is_repealed(articles):
    assert [a["article_number"] for a in articles if a["abrogated"]] == [256]


def test_no_empty_article(articles):
    assert all(a["text"].strip() for a in articles)


def test_every_article_has_a_livre(articles):
    assert all(a["livre"] for a in articles)


def test_chunks_cover_every_article_once(chunks):
    assert {c["article_number"] for c in chunks} == set(range(1, LAST_ARTICLE + 1))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_amendments_are_on_every_chunk_of_their_article(chunks):
    amendments = json.loads((DATA / "amendments.json").read_text(encoding="utf-8"))["amendments"]
    for amd in amendments:
        parts = [c for c in chunks if c["article_number"] == amd["article_number"]]
        assert parts, f"amended article {amd['article_number']} has no chunk"
        for c in parts:  # no chunk may show the old rule without its update
            assert amd["law"] in c["amended_by"]
            assert amd["bulletin_officiel"] in c["text"]


def test_eval_questions_point_to_existing_articles():
    questions = json.loads((DATA / "eval" / "questions.json").read_text(encoding="utf-8"))["questions"]
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert q["relevant"] and all(1 <= n <= LAST_ARTICLE for n in q["relevant"]), q["id"]
