"""French text normalisation for lexical (BM25) search.

    "Le licenciement d'une salariée enceinte" -> ['licenci', 'salari', 'enceint']

Steps: lowercase -> split on non-alphanumerics -> drop stop-words -> Snowball French
stemmer (licenciement / licencié / licencier -> "licenci") -> strip accents.
"""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

_stemmer = snowballstemmer.stemmer("french")
_TOKEN_RE = re.compile(r"[^\W_]+")  # runs of letters (accented included) or digits

# Common French function words, plus words that appear in almost every article
# ("article", "loi", "présente") and therefore carry no signal for retrieval.
STOPWORDS = set("""
a au aux avec ce ces cet cette d dans de des du elle elles en et eux il ils je l la le les leur leurs
lui m ma mais me meme mes moi mon n ne ni nos notre nous on ou par pas pour qu que qui s sa se ses
si son sur t ta te tes toi ton tu un une vos votre vous y c j est sont ete etre avoir ont a
dont ou lequel laquelle lesquels lesquelles auquel duquel cela ceci celui celle ceux celles
tout tous toute toutes autre autres meme memes tel telle tels telles ainsi alors donc or car
article articles loi presente present ci dessus dessous dit dite dits dites susvise susvises
""".split())


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    out = []
    for t in tokens:
        # len > 1 also drops single-digit list markers ("1.", "2.") while keeping numbers like "39".
        if len(t) > 1 and strip_accents(t) not in STOPWORDS:
            # Stem BEFORE removing accents: Snowball's French rules rely on them
            # (salarié/salariée/salariés -> "salari"; accents-first gives salar/salarie).
            out.append(strip_accents(_stemmer.stemWord(t)))
    return out
