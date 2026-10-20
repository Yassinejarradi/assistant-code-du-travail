"""Web interface for the Code du travail assistant — ENSAM Casablanca, Master IA & IoT.

Chat layout in the style of ChatGPT / Claude: centred greeting on an empty chat, user messages
as right-aligned bubbles, answers as plain text with collapsible sources, and a sidebar with
the conversation history.

Run:  streamlit run app.py      (needs Ollama running with qwen3:4b-instruct)
"""

import base64
import html
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from chromadb.api.client import SharedSystemClient

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

from chunking import amendment_note, hierarchy_path, load_amendments  # noqa: E402
from rag import LLM_MODEL, RAG  # noqa: E402

AUTHOR = "Yassine Jarradi"
SCHOOL = "ENSAM Casablanca"
PROGRAM = "Master IA & IoT"
LOGO = ROOT / "assets" / "ensam_logo.png"
DATA = ROOT / "data"
CHUNKS_FILE = DATA / "chunks.json"
BUILD_ID = DATA / "index" / "BUILD_ID"  # rewritten by build_index.py when the index changes
ASSISTANT_AVATAR = "⚖️"

EXAMPLES = [
    "Je suis enceinte, mon employeur peut-il me licencier ?",
    "Combien de jours de congé payé par mois de travail ?",
    "Je suis agent de sécurité, combien d'heures par jour dois-je travailler ?",
    "Comment sont payées les heures supplémentaires ?",
    "Quelle est la durée de la période d'essai pour un cadre ?",
    "Que dit l'article 39 ?",
]

st.set_page_config(page_title="Assistant Code du travail", page_icon="⚖️", layout="centered")

CSS = """
<style>
#MainMenu, footer { visibility: hidden; }
[data-testid="stSkillsNudgeAnchor"] { display: none; }  /* Streamlit's own promo popup */
.block-container, [data-testid="stMainBlockContainer"] { max-width: 780px; padding-top: 2rem; padding-bottom: 7rem; }

/* ---- empty state: centred greeting */
.greet { text-align: center; margin-top: 18vh; }
.greet .icon { width: 56px; height: 56px; margin: 0 auto 1rem; border-radius: 16px; background: #EAF0F7;
               display: flex; align-items: center; justify-content: center; font-size: 1.8rem; }
.greet h1 { font-size: 2rem; font-weight: 600; color: #1B2433; margin: 0 0 .5rem; padding: 0; }
.greet p { color: #6B7686; font-size: 1rem; margin: 0 auto; max-width: 520px; }

/* ---- messages: no boxes; user = right-aligned grey bubble, assistant = plain text */
div[data-testid="stChatMessage"] { background: transparent; padding: .4rem 0; gap: .75rem; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #EDF0F4; border-radius: 20px; padding: .6rem 1.05rem; max-width: 78%; flex: 0 1 auto;
    margin-left: auto !important; margin-right: 0 !important; }  /* Streamlit centres it otherwise */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p { margin: 0; }
[data-testid="stChatMessageAvatarCustom"] { background: #EAF0F7; border-radius: 50%; }
[data-testid="stChatMessageContent"] { font-size: 1rem; line-height: 1.65; }

/* clickable citations: one small button per cited article, opening the full text */
.cite-label { font-size: .8rem; color: #8A94A3; margin: .3rem 0 .1rem; }
div[data-testid="stChatMessage"] [data-testid="stPopover"] button { border-radius: 999px; padding: .1rem .75rem;
    min-height: 0; font-size: .82rem; color: #1F5793; border-color: #C9D6E8; background: #F4F7FB; }
div[data-testid="stChatMessage"] [data-testid="stPopover"] button:hover { border-color: #1F5793; background: #EAF0F7; }
.searched { font-size: .78rem; color: #8A94A3; margin-bottom: .2rem; }

/* sources: compact, discreet */
div[data-testid="stChatMessage"] [data-testid="stExpander"] details { border: none; background: transparent; }
div[data-testid="stChatMessage"] [data-testid="stExpander"] summary { padding: .2rem 0; font-size: .85rem; color: #6B7686; }
.src { border-left: 3px solid #DCE4EF; padding: .2rem 0 .2rem .8rem; margin: .2rem 0 .9rem; }
.src.is-cited { border-left-color: #1F5793; }
.src-head { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; }
.art { font-weight: 600; color: #1F5793; font-size: .92rem; }
.pill { font-size: .7rem; padding: .05rem .5rem; border-radius: 999px; font-weight: 600; }
.cited { background: #E3F4EA; color: #1E7B45; }
.amended { background: #FFF4DB; color: #8A5A00; }
.path { font-size: .75rem; color: #8A94A3; margin: .15rem 0 .3rem; }
.src-text { font-size: .85rem; color: #3A4555; white-space: pre-wrap; max-height: 12rem; overflow-y: auto; }

/* ---- input pinned at the bottom, rounded like ChatGPT, with a disclaimer under it */
[data-testid="stChatInput"] { border-radius: 26px; box-shadow: 0 2px 14px rgba(27,36,51,.08); }
[data-testid="stBottomBlockContainer"] { max-width: 780px; padding-bottom: 1.6rem; }
[data-testid="stBottomBlockContainer"]::after {
    content: "L'assistant peut se tromper : vérifiez les articles cités. Information juridique générale, pas un conseil juridique.";
    display: block; text-align: center; font-size: .74rem; color: #8A94A3; margin-top: .5rem; }

/* ---- sidebar: like a chat-history panel */
section[data-testid="stSidebar"] { background: #F1F4F8; border-right: 1px solid #E3E9F2; }
section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; }
section[data-testid="stSidebar"] .stButton button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.side-title { font-size: .74rem; font-weight: 600; color: #8A94A3; text-transform: uppercase;
              letter-spacing: .04em; margin: 1rem 0 .2rem; }
.side-logo { background: #fff; border-radius: 10px; padding: .5rem; margin-bottom: .8rem; }
.side-logo img { width: 100%; display: block; }
.author { border-top: 1px solid #DCE4EF; margin-top: 1.4rem; padding-top: .9rem; font-size: .82rem; color: #6B7686; }
.author b { color: #1B2433; font-size: .92rem; }
.steps { font-size: .82rem; color: #3A4555; padding-left: 1.1rem; margin: 0; }
.steps li { margin-bottom: .35rem; }
</style>
"""


def data_version() -> tuple[float, float]:
    """Changes only when the chunks or the index are rebuilt. (Chroma's own files can't be used:
    their mtimes change on every read-only query, which made the app reload after each question.)"""
    return CHUNKS_FILE.stat().st_mtime, BUILD_ID.stat().st_mtime if BUILD_ID.exists() else 0.0


@st.cache_resource(show_spinner="Démarrage : chargement des modèles (≈1 min, une seule fois)…",
                   max_entries=1)
def load_rag(version: tuple[float, float]) -> RAG:
    # Cached: runs once per server, not on every question. `version` (see data_version) is part
    # of the cache key, so after the chunks or the index are updated (e.g. a new amendment) the
    # next page load rebuilds the RAG instead of serving stale articles. max_entries=1 frees the
    # previous copy.
    # Chroma keeps one in-memory copy of the index per process and doesn't see updates written
    # by another process (e.g. `build_index.py --update`): drop it so the index is re-read.
    SharedSystemClient.clear_system_cache()
    rag = RAG()
    try:
        rag.warm_up()
    except Exception:
        pass  # Ollama not running yet: the question handler below shows a clear error
    return rag


@st.cache_data
def articles_by_number(version: tuple[float, float]) -> dict[int, dict]:
    """Full text of every article (+ amendment notes), for the citation popovers. Cited articles
    may not be among the retrieved chunks (e.g. a cross-reference), so read articles.json."""
    amendments = load_amendments()
    out = {}
    for a in json.loads((DATA / "articles.json").read_text(encoding="utf-8")):
        notes = [amendment_note(x) for x in amendments.get(a["article_number"], [])]
        out[a["article_number"]] = {"path": hierarchy_path(a).replace(" > ", " › "),
                                    "text": "\n".join([a["text"], *notes]),
                                    "amended_by": [x["law"] for x in amendments.get(a["article_number"], [])]}
    return out


@st.cache_data
def logo_b64() -> str:
    return base64.b64encode(LOGO.read_bytes()).decode() if LOGO.exists() else ""


# ---- conversations (kept for the browser session, like a chat history)
def new_conversation() -> None:
    cid = uuid.uuid4().hex
    st.session_state.convs[cid] = {"title": "Nouvelle conversation", "messages": []}
    st.session_state.current = cid


def current_messages() -> list[dict]:
    return st.session_state.convs[st.session_state.current]["messages"]


def ask(question: str) -> None:
    """Queue a question (from an example) and reload the page to answer it."""
    st.session_state.pending = question


if "convs" not in st.session_state:
    st.session_state.convs = {}
    new_conversation()


def show_citations(cited: list[int]) -> None:
    """One button per cited article; clicking it opens the full article text."""
    articles = articles_by_number(data_version())
    cited = [n for n in cited if n in articles]
    if not cited:
        return
    st.markdown('<div class="cite-label">Articles cités — cliquez pour lire le texte :</div>',
                unsafe_allow_html=True)
    with st.container(horizontal=True, gap="small"):
        for n in cited:
            art = articles[n]
            with st.popover(f"Art. {n}", width="content"):
                st.markdown(f"**Article {n}**" + ("  ·  *modifié par la loi n° 32.26*" if art["amended_by"] else ""))
                st.caption(art["path"])
                st.markdown(f'<div class="src-text">{html.escape(art["text"])}</div>', unsafe_allow_html=True)


def show_sources(sources: list[dict], unsupported: list[int]) -> None:
    with st.expander(f"Tous les articles consultés ({len(sources)})"):
        for s in sources:
            pills = '<span class="pill cited">cité</span>' if s["cited"] else ""
            pills += "".join(f'<span class="pill amended">Modifié · {html.escape(law.split(" modifiant")[0])}</span>'
                             for law in s.get("amended_by", []))
            st.markdown(
                f'<div class="src{" is-cited" if s["cited"] else ""}"><div class="src-head">'
                f'<span class="art">Article {s["number"]}</span>{pills}</div>'
                f'<div class="path">{html.escape(s["path"])}</div>'
                f'<div class="src-text">{html.escape(s["text"])}</div></div>',
                unsafe_allow_html=True)
    if unsupported:
        st.warning(f"Articles cités absents des sources consultées : {unsupported}. À vérifier.")


st.markdown(CSS, unsafe_allow_html=True)

# Read the question FIRST (the input stays pinned at the bottom wherever it is called), so the new
# message and the conversation title exist before the greeting and the sidebar history are drawn.
question = st.chat_input("Posez votre question sur le Code du travail…")
question = question or st.session_state.pop("pending", None)
if question:
    conv = st.session_state.convs[st.session_state.current]
    if not conv["messages"]:  # first question names the conversation in the sidebar
        conv["title"] = question if len(question) <= 42 else question[:40].rstrip() + "…"
    conv["messages"].append({"role": "user", "content": question})

# ---- sidebar
with st.sidebar:
    if logo_b64():
        st.markdown(f'<div class="side-logo"><img src="data:image/png;base64,{logo_b64()}" '
                    f'alt="ENSAM Casablanca"></div>', unsafe_allow_html=True)
    if st.button("＋  Nouvelle conversation", width="stretch"):
        if current_messages():  # don't stack empty conversations
            new_conversation()
        st.rerun()

    history = [(cid, c) for cid, c in st.session_state.convs.items() if c["messages"]]
    if history:
        st.markdown('<div class="side-title">Récentes</div>', unsafe_allow_html=True)
        for cid, conv in reversed(history):
            is_current = cid == st.session_state.current
            if st.button(conv["title"], key=f"conv-{cid}", width="stretch",
                         type="secondary" if is_current else "tertiary"):
                st.session_state.current = cid
                st.rerun()

    st.markdown('<div class="side-title">Aide</div>', unsafe_allow_html=True)
    with st.expander("💡 Exemples de questions"):
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", width="stretch", type="tertiary", on_click=ask, args=(ex,))
    with st.expander("ℹ️ Comment ça marche"):
        st.markdown(
            '<ol class="steps">'
            "<li><b>Recherche</b> des 5 articles les plus pertinents parmi les 589 du Code "
            "(embeddings <i>bge-m3</i> + base vectorielle Chroma)</li>"
            f"<li><b>Rédaction</b> de la réponse par un LLM local (<i>{LLM_MODEL}</i>, Ollama), "
            "uniquement à partir de ces articles</li>"
            "<li><b>Vérification</b> que chaque article cité figure dans les sources</li></ol>",
            unsafe_allow_html=True)
        st.caption("100 % local : aucune donnée ne quitte l'ordinateur. Source : loi n° 65-99, "
                   "version consolidée du 26/10/2011, mise à jour par la loi n° 32.26 (2026).")

    st.markdown(f'<div class="author">Réalisé par<br><b>{AUTHOR}</b><br>{PROGRAM} · {SCHOOL}</div>',
                unsafe_allow_html=True)

rag = load_rag(data_version())  # warm up as soon as the page opens
messages = current_messages()

# ---- empty conversation: centred greeting (no suggestions on the main page)
if not messages:
    st.markdown(
        f'<div class="greet"><div class="icon">{ASSISTANT_AVATAR}</div>'
        "<h1>Comment puis-je vous aider ?</h1>"
        "<p>Posez une question sur le Code du travail marocain : je réponds en citant les "
        "articles de la loi n° 65-99.</p></div>", unsafe_allow_html=True)

# ---- conversation
for msg in messages:
    with st.chat_message(msg["role"], avatar=ASSISTANT_AVATAR if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("searched"):
                st.markdown(f'<div class="searched">🔎 Recherche : {html.escape(msg["searched"])}</div>',
                            unsafe_allow_html=True)
            if msg.get("cited"):  # articles only when the answer actually relies on them
                show_citations(msg["cited"])
                show_sources(msg["sources"], msg["unsupported"])

if question:  # the user message is already displayed above; now write the answer
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # previous (question, answer) pairs of this conversation, for follow-up questions
        history = [(q["content"], a["content"]) for q, a in zip(messages[:-1:2], messages[1:-1:2])]
        try:
            legal = rag.is_legal(question, history)  # "Bonjour", "merci"… -> natural reply, no search
        except Exception:
            legal = True
        if not legal:
            try:
                text = st.write_stream(rag.chat(question, history))
            except Exception as e:
                st.error(f"Le modèle de langage ne répond pas ({e}). Lancez Ollama (`ollama serve`).")
                st.stop()
            messages.append({"role": "assistant", "content": text.strip(), "searched": None, "cited": [],
                             "sources": [], "unsupported": []})
            st.stop()
        try:
            # follow-ups only refer to legal exchanges ("Bonjour" is not context for rewriting)
            legal_history = [(q["content"], a["content"])
                             for q, a in zip(messages[:-1:2], messages[1:-1:2]) if a.get("sources")]
            with st.spinner("Recherche dans le Code du travail…"):
                standalone = rag.condense(question, legal_history)  # "Et les concierges ?" -> full question
                hits = rag.retrieve(standalone)
            searched = standalone if standalone != question else None
            if searched:
                st.markdown(f'<div class="searched">🔎 Recherche : {html.escape(searched)}</div>',
                            unsafe_allow_html=True)
            text = st.write_stream(rag.stream(standalone, hits))
        except Exception as e:  # Ollama not running, model missing, ...
            st.error(f"Le modèle de langage ne répond pas ({e}). Lancez Ollama (`ollama serve`) "
                     f"et vérifiez que le modèle est installé (`ollama pull {LLM_MODEL}`).")
            st.stop()
        ans = rag.check(standalone, text, hits)
        sources = [{
            "number": h.article_number,
            "cited": h.article_number in ans.cited,
            "path": " › ".join(x for x in (h.chunk["livre"], h.chunk["titre"], h.chunk["chapitre"],
                                           h.chunk["section"]) if x),
            "text": h.chunk["text"],
            "amended_by": h.chunk.get("amended_by", []),
        } for h in hits]
        if ans.cited:
            show_citations(ans.cited)
            show_sources(sources, ans.unsupported)

    messages.append({"role": "assistant", "content": ans.text, "searched": searched, "cited": ans.cited,
                     "sources": sources, "unsupported": ans.unsupported})
