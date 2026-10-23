"""Rule-based parts of the RAG pipeline: citation parsing and the small-talk / legal router."""

import pytest

from rag import LEGAL_HINT_RE, SMALL_TALK_RE, extract_citations


@pytest.mark.parametrize("text, expected", [
    ("(Article 159)", [159]),
    ("(Article 231, 1)", [231]),                       # point 1 of art. 231, not article 1
    ("prévue aux articles 154 et 156 ci-dessus", [154, 156]),
    ("les articles 152 à 163", [152, 163]),
    ("articles 274, 275 et 277", [274, 275, 277]),
    ("Article 193 et Article 188", [188, 193]),
    ("conformément à l'article 39.", [39]),
    ("Les articles trouvés ne permettent pas de répondre à cette question.", []),
])
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


@pytest.mark.parametrize("message", [
    "Bonjour", "bonjour !", "Salut", "Bonsoir", "Merci beaucoup !", "ok merci", "Merci pour votre aide",
    "Qui es-tu ?", "Au revoir", "ça va ?", "salam",
])
def test_small_talk_is_recognised_by_rules(message):
    assert SMALL_TALK_RE.match(message)


@pytest.mark.parametrize("message", [
    "Quelle est la durée du congé de maternité ?",
    "Bonjour, combien de jours de congé payé par mois ?",   # greeting + real question
    "Merci, et pour le père ?",
    "Que dit l'article 39 ?",
])
def test_questions_are_not_small_talk(message):
    assert not SMALL_TALK_RE.match(message)


@pytest.mark.parametrize("message", [
    "Mon chef me crie dessus tous les jours, c'est normal ?",
    "On m'a viré sans raison",
    "Je n'ai pas reçu ma paie ce mois-ci",
    "Est-ce que je peux refuser de travailler le dimanche ?",
    "Que dit l'article 39 ?",
])
def test_workplace_situations_are_detected_as_legal(message):
    assert LEGAL_HINT_RE.search(message)
