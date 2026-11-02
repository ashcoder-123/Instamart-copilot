"""Part D (tasks 1-4): policy chunking, TF-IDF retrieval, grounded answers, evaluation.
Run from the repository root:  python src/part_d_rag.py"""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402

POLICY_DIR = "data/policies"

RAG_SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer the question using only the supplied "
              "context. If the context does not contain the answer, say that you could not find the information.")


# ------------------------------------------------------------------ 1. chunking
def chunk_words(words, size=40, overlap=10):
    """First chunk covers words 0..size-1, the next starts size-overlap words later; stop once a chunk reaches the end."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_chunks(size=40, overlap=10, policy_dir=POLICY_DIR):
    out = []
    for fn in sorted(f for f in os.listdir(policy_dir) if f.endswith(".txt")):
        words = open(os.path.join(policy_dir, fn), encoding="utf-8").read().split()
        out += [{"source": fn, "text": c} for c in chunk_words(words, size, overlap)]
    return out


# ------------------------------------------------------------------ 2. TF-IDF retrieval
class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer().fit([c["text"] for c in chunks])
        self.matrix = self.vectorizer.transform([c["text"] for c in chunks])


_default = {}


def default_index():
    if "idx" not in _default:
        _default["idx"] = Index(load_chunks(40, 10))
    return _default["idx"]


def retrieve(query, k=3, index=None):
    """Return the k best chunks as dicts with source, score, text (cosine similarity on TF-IDF vectors)."""
    idx = index or default_index()
    sims = cosine_similarity(idx.vectorizer.transform([query]), idx.matrix)[0]
    order = sims.argsort()[::-1][:k]
    return [{"source": idx.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": idx.chunks[i]["text"]}
            for i in order]


# ------------------------------------------------------------------ 3. augment and answer
def answer(question, index=None):
    hits = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]


# ------------------------------------------------------------------ 4. evaluation
def evaluate(label, index):
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = n_ans = n_unans = 0
    failures = []
    for r in qa.itertuples(index=False):
        text, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            n_unans += 1
            ok = text.startswith("I could not find")
            fallback += ok
            if not ok:
                failures.append((r.question, "expected fallback", sources, text))
        else:
            n_ans += 1
            d, f = r.expected_doc in sources, r.expected_fact in text
            doc_hit += d
            fact_hit += f
            if not (d and f):
                failures.append((r.question, f"doc_in_top3={d} fact_in_answer={f} (expected '{r.expected_fact}')", sources, text))
    print(f"\n=== {label}: {len(index.chunks)} chunks ===")
    print(f"answerable: expected_doc in top 3 = {doc_hit}/{n_ans} | expected_fact in answer = {fact_hit}/{n_ans}")
    print(f"unanswerable: fallback answers = {fallback}/{n_unans}")
    for q, why, src, text in failures:
        print(f"  FAIL: {q}\n        {why}\n        sources={src}\n        answer={text}")
    return failures


if __name__ == "__main__":
    chunks = load_chunks(40, 10)
    print("chunks total:", len(chunks), "|", pd.Series([c["source"] for c in chunks]).value_counts().sort_index().to_dict())
    for h in retrieve("refund policy late delivery"):
        print(f"  top hit: {h['source']} score {h['score']}")
    evaluate("overlap 10", Index(chunks))
    evaluate("overlap 0", Index(load_chunks(40, 0)))
