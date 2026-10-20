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

# ---- routing: small talk vs legal question
# Obvious small talk is recognised by rules (instant); anything else goes to a quick LLM router.
SMALL_TALK_RE = re.compile(
    r"^\W*(bonjour|bonsoir|salut|coucou|hello|hey|hi|salam|salam alaykoum|slt|cc|"
    r"merci( beaucoup| bien)?|thanks?|au revoir|bye|à bientôt|bonne (journée|soirée)|"
    r"(ça|ca) va|comment (ça|ca) va|ok|okay|d'accord|super|parfait|top|génial|"
    r"qui es[- ]tu|tu es qui|t'es qui|que (sais|peux)[- ]tu faire|comment tu t'appelles)"
    r"([\s,!.?]+(bonjour|merci|beaucoup|à toi|à vous|encore|pour (ta|votre) (aide|réponse)))*\W*$",
    re.IGNORECASE)
LEGAL_HINT_RE = re.compile(
    r"\b(article|art\.|code|loi|employeur|patron|salari|travail|contrat|licenci|cong[ée]|salaire|"
    r"heures?|préavis|indemnit|grève|syndic|délégu|retraite|démission|essai|maternit|paternit|"
    r"accident|inspect|prime|smig|cdd|cdi|embauch|stagiaire|apprenti|chef|boss|patronne|coll[èe]gue|"
    r"harc[eè]l|insult|vir[ée]|renvoy|pay[ée]|paie|horaire|dimanche|f[ée]ri[ée]|vacances|maladie|"
    r"enceinte|grossesse|bureau|entreprise|usine|société|poste|mutation|sanction|faute)", re.IGNORECASE)

ROUTER_PROMPT = """Tu classes les messages envoyés à un assistant spécialisé dans le Code du travail marocain.
{history}Message : {question}

Ce message est-il une question ou une demande sur le droit du travail, une situation vécue au travail ou un article de loi (y compris une question de suivi comme « et pour le père ? » ou une situation comme « mon chef ne me paie pas ») ?
JURIDIQUE = question ou situation liée au travail. CONVERSATION = salutation, remerciement, question sur l'assistant, ou sujet sans rapport avec le travail.
En cas de doute, réponds JURIDIQUE.
Réponds par un seul mot : JURIDIQUE ou CONVERSATION."""

CHAT_PROMPT = """Tu es l'Assistant Code du travail, un assistant chaleureux et naturel qui aide les salariés et les employeurs à comprendre le Code du travail marocain (loi n° 65-99). Tu as été réalisé par Yassine Jarradi (Master IA & IoT, ENSAM Casablanca).

Le message de l'utilisateur n'est pas une question juridique : réponds comme une personne, en une à trois phrases, en français, en vouvoyant l'utilisateur (sauf s'il te tutoie). Au plus un emoji simple, jamais de drapeau.
- Salutation : salue en retour et propose ton aide sur le droit du travail.
- Remerciement ou au revoir : réponds poliment, propose de revenir si besoin.
- Question sur toi : explique brièvement ce que tu fais (tu réponds aux questions sur le Code du travail en citant les articles de loi).
- Sujet sans rapport avec le travail : ne réponds pas à la question ; dis gentiment que tu es spécialisé dans le Code du travail marocain et propose un exemple de question dans ce domaine.
Ne cite aucun article et ne donne aucune information juridique dans ce cas."""

# Few-shot examples: a 4B model follows the style it is shown much better than style rules
# (with rules only it used "tu", several emojis, and answered off-topic questions).
CHAT_EXAMPLES = [
    ("Bonjour", "Bonjour ! Je suis l'Assistant Code du travail. Comment puis-je vous aider ? Vous pouvez "
                "par exemple me demander combien de jours de congé payé vous avez par mois."),
    ("Quelle est la capitale de l'Espagne ?", "Je suis spécialisé dans le Code du travail marocain, je ne "
                "peux donc pas vous répondre sur ce sujet. En revanche, je peux vous aider sur vos droits au "
                "travail : congés, salaire, licenciement, horaires…"),
    ("Merci !", "Avec plaisir ! N'hésitez pas si vous avez d'autres questions sur le droit du travail. 🙂"),
]

FLAG_RE = re.compile("[🇦-🇿]")  # flag emojis render as letters ("MA") on Windows

# Singular "Article N" cites one article ("Article 231, 1" = point 1 of art. 231, not art. 1);
# only the plural introduces a list ("articles 154 et 156", "articles 152 à 163").
CITATION_RE = re.compile(r"\bArticle\s+(\d{1,3})\b|\bArticles\s+(\d{1,3})((?:\s*(?:,|et|à)\s*\d{1,3})*)",
                         re.IGNORECASE)


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
        if m.group(1):
            numbers.append(int(m.group(1)))
        else:
            numbers.append(int(m.group(2)))
            numbers += [int(n) for n in re.findall(r"\d{1,3}", m.group(3))]
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

    def is_legal(self, question: str, history: list[tuple[str, str]]) -> bool:
        """Does this message need the article search? Greetings, thanks, questions about the bot
        and off-topic chat don't: answering them with 5 random articles looks robotic."""
        if SMALL_TALK_RE.match(question):
            return False
        if LEGAL_HINT_RE.search(question):
            return True
        last = f"Question précédente : {history[-1][0]}\n" if history else ""
        response = ollama.generate(model=self.llm_model, keep_alive=KEEP_ALIVE,
                                   prompt=ROUTER_PROMPT.format(history=last, question=question),
                                   options={"temperature": 0, "num_predict": 4})
        return "CONVERSATION" not in response["response"].upper()  # unclear output -> legal (safer)

    def chat(self, question: str, history: list[tuple[str, str]]) -> Iterator[str]:
        """Natural reply to small talk, without retrieval or citations."""
        messages = [{"role": "system", "content": CHAT_PROMPT}]
        for q, a in CHAT_EXAMPLES:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        for q, a in history[-2:]:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a[:400]}]
        messages.append({"role": "user", "content": question})
        for chunk in ollama.chat(model=self.llm_model, messages=messages, stream=True,
                                 keep_alive=KEEP_ALIVE, options={"temperature": 0.4}):
            yield FLAG_RE.sub("", chunk["message"]["content"])

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

    def answer(self, question: str, history: list[tuple[str, str]] | None = None,
               echo: bool = False) -> Answer:
        history = history or []
        legal = self.is_legal(question, history)
        if legal:  # rewrite follow-ups using legal exchanges only ("Bonjour" is not context)
            question = self.condense(question, [(q, a) for q, a in history if extract_citations(a)])
        hits = self.retrieve(question) if legal else []
        parts = []
        for piece in (self.stream(question, hits) if legal else self.chat(question, history)):
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
    history: list[tuple[str, str]] = []  # interactive mode remembers the conversation (follow-ups)
    for q in questions:
        print()
        try:
            ans = rag.answer(q, history, echo=True)
        except ollama.ResponseError as e:
            raise SystemExit(f"Ollama error: {e}. Is the model pulled? `ollama pull {args.model}`")
        except ConnectionError:
            raise SystemExit("Cannot reach Ollama. Start it with `ollama serve` (or the Ollama app).")
        history.append((q, ans.text))
        if ans.cited:  # small talk has no sources to show
            print_sources(ans)


if __name__ == "__main__":
    main()
