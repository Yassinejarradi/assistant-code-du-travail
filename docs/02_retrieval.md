# Step 2 — Retrieval: design, experiments and results

Notes written while building the retriever. They explain *why* each choice was made, and the
numbers can be reused in the thesis.

## 1. Pipeline

```
articles.json ─► chunking.py ─► chunks.json ─┬─► build_index.py: bge-m3 ─► Chroma (604 × 1024)
                                             └─► BM25 (built in memory at query time)

question ─► retriever.py ─► direct lookup ("article 39")  ┐
                         ├─► dense search (cosine, Chroma) ├─► weighted RRF ─► top-k articles
                         └─► BM25 (hybrid mode only)       ┘
```

## 2. Chunking

| Choice | Reason |
|---|---|
| 1 article = 1 chunk | The article is the citation unit, and 95% of articles are ≤ 1,070 chars (~270 tokens) |
| Split above 1,500 chars, at paragraph boundaries | 12 articles (e.g. 586, 533, 462). Very long chunks dilute the embedding |
| Each part repeats the article's first paragraph | Lead-ins such as "Sont punis d'une amende de … :" give meaning to the list items that follow |
| Embedded text = hierarchy path + "Article N" + text | Chapter titles add the topic ("protection de la maternité") even when the body doesn't name it |
| Results collapsed per article | Two chunks of the same article must not take two of the k slots |

Result: 589 articles → 604 chunks.

## 3. The retrievers

**BM25** is lexical: it scores word overlap, weighted by rarity (IDF), with term-frequency
saturation and length normalisation. French preprocessing: lowercase → stop-words → Snowball
stemmer → strip accents.
*Lesson learned:* stemming must happen **before** stripping accents. Otherwise
salarié/salariée/salariés give `salar`/`salarie`/`salar` instead of a single stem `salari`.

**Dense** retrieval uses `BAAI/bge-m3` (568M parameters, multilingual, 1,024 dimensions,
normalised vectors so cosine = dot product). Chroma stores the vectors and searches them with
HNSW, an approximate nearest-neighbour graph. At 604 vectors it is effectively exact.

**Direct lookup**: a regex catches "article 39", "art. 39" and "article premier", and the cited
article is forced to rank 1.

**Fusion**: Reciprocal Rank Fusion (Cormack et al., 2009), `score = Σ wᵢ / (60 + rankᵢ)`.
It uses only ranks, so it needs no calibration between BM25 scores and cosine similarities.

## 4. Evaluation set

`data/eval/questions.json` has 50 questions, each with its relevant article(s), in 4 styles:
formal (18), colloquial (24), vocab_gap (6: words absent from the law, such as "SMIG",
"intérim", "heures sup"), and reference (2).
Metrics: **Recall@k** (is a relevant article in the top k?) and **MRR** (mean of 1/rank).

⚠ The questions were drafted by Claude from the article texts and still need human review.

## 5. Results (bge-m3, k counted in articles)

| Mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.42 | 0.62 | 0.64 | 0.68 | 0.52 |
| **Dense (default)** | **0.74** | **0.88** | **0.92** | **0.92** | **0.81** |
| Hybrid, RRF equal weights | 0.58 | 0.74 | 0.78 | 0.86 | 0.67 |

By style, R@5:

| Style | BM25 | Dense | Hybrid (w=1) |
|---|---|---|---|
| formal | 0.89 | 1.00 | 1.00 |
| colloquial | 0.50 | 0.83 | 0.67 |
| vocab_gap | 0.33 | 1.00 | 0.50 |

BM25 weight sweep in hybrid mode:

| BM25 weight | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|
| R@5 | 0.92 | 0.90 | 0.90 | 0.88 | 0.82 | 0.82 | 0.78 |
| MRR | 0.81 | 0.81 | 0.78 | 0.75 | 0.70 | 0.68 | 0.67 |

## 6. Findings

1. **Dense retrieval dominates** on natural questions, because users don't speak like the law
   ("enceinte" vs "état de grossesse", "SMIG" vs "salaire minimum légal").
2. **"Hybrid always helps" did not hold here.** With RRF, a weak retriever drags down a strong
   one: BM25 pushes generic "licenciement" articles above art. 159. Hybrid search pays off when
   both retrievers are of comparable quality, or on exact-term queries that this set barely
   covers.
3. **Decision:** the default is dense + direct lookup; hybrid stays available with weight 0.1.
4. **Caveats:** with 50 questions, one question equals 2 points of recall, and choosing settings
   on the test set itself is optimistic. Validate on new questions written by a human.
5. **Remaining dense misses** (top 5): q06 (union-membership discrimination → syndicate
   articles), q25 (232 vs neighbour 234), q32 (385 vs 386–387), q36 (304 vs 306). Three of the
   four are *neighbouring articles in the same section*. Possible next improvements:
   - a cross-encoder **reranker** (e.g. `BAAI/bge-reranker-v2-m3`) over the top 20;
   - **neighbour expansion** (also give the LLM articles n−1 and n+1).

## 7. Reproduce

```bash
python src/chunking.py
python src/build_index.py            # ~10 min on CPU, first run downloads bge-m3 (~2.3 GB)
python src/evaluate_retrieval.py --show-misses
python src/retriever.py "Je suis enceinte, mon employeur peut-il me licencier ?"
```
