"""Answer questions about the Code du travail with retrieved articles + a local LLM (Ollama).

    question ─► Retriever (top-k articles) ─► prompt (rules + articles + question) ─► LLM ─► answer
                                                                       citation check ◄─┘

Requires the Ollama server running locally (`ollama serve`) with the model pulled
(`ollama pull qwen3:4b-instruct`).

Usage:
    python src/rag.py "Quelle est la durée du congé de maternité ?"
    python src/rag.py            # interactive mode
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import ollama

from retriever import Hit, Retriever

# Instruct (non-thinking) variant: the plain "qwen3:4b" tag writes a long hidden reasoning
# before answering and ignores think=False, which made answers take minutes on this laptop.
LLM_MODEL = "qwen3:4b-instruct"
TOP_K = 5
TEMPERATURE = 0.1  # low: we want faithful restatement of the law, not creativity
# Ollama unloads an idle model after 5 min by default, and reloading it costs ~30 s on this laptop.
# Keep it in memory longer so a demo session stays responsive (it is freed automatically after).
KEEP_ALIVE = "30m"

SYSTEM_PROMPT = """Tu es un assistant juridique spécialisé dans le Code du travail marocain (loi n° 65-99).

Règles :
1. Réponds UNIQUEMENT à partir des articles fournis dans le contexte. N'utilise aucune autre connaissance.
2. Cite l'article sur lequel repose chaque affirmation, sous la forme (Article N).
3. Si les articles fournis ne permettent pas de répondre, dis-le clairement : « Les articles trouvés ne permettent pas de répondre à cette question. »
4. Reprends les chiffres, durées et montants exactement comme dans l'article, sans conversion ni calcul (ne transforme pas des semaines en mois, par exemple).
5. Réponds en français, de manière claire et concise, compréhensible par un salarié ou un employeur non juriste.
6. Ne donne pas de conseil personnalisé ; tu expliques ce que dit la loi."""

CONDENSE_PROMPT = """Voici la fin d'une conversation sur le Code du travail marocain, puis une nouvelle question.

{history}

Nouvelle question : {question}

Réécris la nouvelle question pour qu'elle soit compréhensible seule, sans la conversation (remplace « et », « ça », « dans ce cas », « eux »… par ce qu'ils désignent).
Si la nouvelle question est déjà compréhensible seule ou porte sur un autre sujet, recopie-la telle quelle.
Réponds uniquement par la question réécrite, sans explication."""

CITATION_RE = re.compile(r"\bArticles?\s+(\d{1,3})((?:\s*(?:,|et|à)\s*\d{1,3})*)", re.IGNORECASE)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]
    cited: list[int]            # article numbers cited in the answer
    unsupported: list[int]      # cited but absent from the context given to the LLM (possible hallucination)


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        c = h.chunk
        where = " > ".join(x for x in (c["titre"], c["chapitre"], c["section"]) if x)
        blocks.append(f"[Article {h.article_number}] ({where})\n{c['text']}")
    return "\n\n".join(blocks)


def extract_citations(text: str) -> list[int]:
    numbers = []
    for m in CITATION_RE.finditer(text):
        numbers.append(int(m.group(1)))
        numbers += [int(n) for n in re.findall(r"\d{1,3}", m.group(2))]
    return sorted(set(numbers))


class RAG:
    def __init__(self, llm_model: str = LLM_MODEL, k: int = TOP_K):
        self.retriever = Retriever()
        self.llm_model = llm_model
        self.k = k

    def warm_up(self) -> None:
        """Pay the one-time loading costs up front (measured on this laptop: bge-m3 ~25 s,
        LLM onto the GPU ~30 s, first index read up to a minute), instead of on the first question."""
        self.retrieve("congé annuel")  # loads bge-m3 + opens the Chroma index
        ollama.generate(model=self.llm_model, prompt="", keep_alive=KEEP_ALIVE)  # empty prompt = load only

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Turn a follow-up ("Et les concierges ?") into a standalone question using the last
        exchange, so retrieval knows what it is about. Without history the question is unchanged."""
        if not history:
            return question
        last_q, last_a = history[-1]
        text = f"Question précédente : {last_q}\nRéponse précédente : {last_a[:500]}"
        response = ollama.generate(model=self.llm_model, keep_alive=KEEP_ALIVE,
                                   prompt=CONDENSE_PROMPT.format(history=text, question=question),
                                   options={"temperature": 0, "num_predict": 80})
        raw = response["response"].strip()
        rewritten = raw.splitlines()[0].strip().strip('"«» ') if raw else ""
        return rewritten or question

    # The three stages are separate methods so a UI can show sources first and stream the answer.
    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(question, k=self.k)

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        """Yield the answer piece by piece as the LLM generates it."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Articles du Code du travail :\n\n{build_context(hits)}\n\n"
                                        f"Question : {question}"},
        ]
        for chunk in ollama.chat(model=self.llm_model, messages=messages, stream=True,
                                 keep_alive=KEEP_ALIVE, options={"temperature": TEMPERATURE}):
            yield chunk["message"]["content"]

    @staticmethod
    def check(question: str, text: str, hits: list[Hit]) -> Answer:
        cited = extract_citations(text)
        # A citation is supported if the article was retrieved, or if a retrieved article refers
        # to it (e.g. art. 159 mentions "articles 154 et 156"). Anything else may be invented.
        in_context = {h.article_number for h in hits}
        in_context |= {n for h in hits for n in extract_citations(h.chunk["text"])}
        return Answer(question, text.strip(), hits, cited, [n for n in cited if n not in in_context])

    def answer(self, question: str, echo: bool = False) -> Answer:
        hits = self.retrieve(question)
        parts = []
        for piece in self.stream(question, hits):
            if echo:
                print(piece, end="", flush=True)
            parts.append(piece)
        if echo:
            print()
        return self.check(question, "".join(parts), hits)


def print_sources(ans: Answer) -> None:
    print("\nSources :")
    for h in ans.sources:
        mark = "✓ cité" if h.article_number in ans.cited else "  "
        print(f"  {mark:7} Article {h.article_number} — {h.chunk['text'][:90].replace(chr(10), ' ')}…")
    if ans.unsupported:
        print(f"  ⚠ articles cités absents du contexte fourni au modèle (à vérifier) : {ans.unsupported}")
    print("\nℹ Information juridique générale, pas un conseil juridique. Source : loi n° 65-99, "
          "version consolidée du 26/10/2011.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    rag = RAG(args.model, args.k)
    questions = [args.question] if args.question else iter(lambda: input("\nQuestion (vide pour quitter) : ").strip(), "")
    for q in questions:
        print()
        try:
            ans = rag.answer(q, echo=True)
        except ollama.ResponseError as e:
            raise SystemExit(f"Ollama error: {e}. Is the model pulled? `ollama pull {args.model}`")
        except ConnectionError:
            raise SystemExit("Cannot reach Ollama. Start it with `ollama serve` (or the Ollama app).")
        print_sources(ans)


if __name__ == "__main__":
    main()
