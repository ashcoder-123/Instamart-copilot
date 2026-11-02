"""Part C: support-ticket triage with guardrails, redaction, retries, validation, token budget and routing.
Run after part_b_segments.py:  python src/part_c_triage.py"""
import json
import os
import random
import re
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402
from mock_llm import TransientLLMError, est_tokens  # noqa: E402

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

# ------------------------------------------------------------------ prompts
ZERO_SHOT = ('You are a support ticket router. CLASSIFY the ticket into exactly one category: '
             'billing, delivery, technical or fraud. '
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}, and nothing else.')
FEW_SHOT = ZERO_SHOT + (
    '\nEXAMPLES:'
    '\nTicket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}'
    '\nTicket: Someone used my wallet without permission, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}'
    '\nTicket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}'
    '\nTicket: My order has not arrived yet. -> {"category": "delivery", "reason": "late order"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT


def redact(text, counts=None):
    for name, pat, repl in PATTERNS:               # email, then phone, then order id
        text, n = pat.subn(repl, text)
        if counts is not None:
            counts[name] += n
    return text


def has_pii(text):
    return any(p.search(text) for _, p, _ in PATTERNS)


def call_with_retry(messages, ticket_id, log, stats):
    """Up to MAX_RETRIES retries; wait before retry n is BASE_DELAY * 2**n * uniform(0.9, 1.15)."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient_errors"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            assert BASE_DELAY * 2 ** n * 0.9 <= wait <= BASE_DELAY * 2 ** n * 1.15
            log.append(f"retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s")
            time.sleep(wait)


def validate(text):
    """Return (status, category). Never guesses."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        found = {w for w in words if w in CATEGORIES}
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", ""
    if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""


def run_pipeline(tickets, system_prompt, token_budget=None, seed=42):
    mock_llm.reset()                                # reset immediately before each full run
    random.seed(seed)
    log, sent = [], []
    stats = {"transient_errors": 0, "permanent_errors": 0, "total_tokens": 0}
    rows = []
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0}
        text = t.text
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and stats["total_tokens"] >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            assert not any(has_pii(m["content"]) for m in messages), "PII reached a prompt"
            sent.append(messages)
            try:
                resp = call_with_retry(messages, t.ticket_id, log, stats)
            except TransientLLMError:
                stats["permanent_errors"] += 1
                row["status"] = "flagged_invalid_output"
            else:
                row["tokens_used"] = resp.usage.total_tokens
                stats["total_tokens"] += resp.usage.total_tokens
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), stats, log, sent


def route(df, seg):
    out = df.copy()
    seg = seg.set_index("customer_id")
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    premium = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)
    human = out["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    out.loc[premium & ~human, ["route", "priority"]] = ["premium_desk", "HIGH"]
    out.loc[(out["category"] == "fraud") & ~human, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def summarise(name, df, stats, log):
    classified = df[df["status"].isin(["classified_json", "classified_keyword"])]
    truth = tickets.set_index("ticket_id").loc[classified["ticket_id"], "true_category"].values
    correct = int((classified["category"].values == truth).sum())
    print(f"\n=== {name} ===")
    print("status counts:", df["status"].value_counts().to_dict())
    print(f"transient errors: {stats['transient_errors']} | retry log lines: {len(log)} | permanent errors: {stats['permanent_errors']}")
    print(f"classified: {len(classified)} | accuracy: {correct}/{len(classified)} = {correct / max(len(classified), 1) * 100:.1f}%")
    print(f"total tokens: {stats['total_tokens']} | tokens per classified ticket: {stats['total_tokens'] / max(len(classified), 1):.2f}")
    return classified, truth


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    seg = pd.read_csv("outputs/customer_segments.csv")
    print("tickets:", len(tickets))

    # --- guardrail figures and redaction counts over all 164 tickets
    empty = int((~tickets["text"].str.strip().astype(bool)).sum())
    long_ = tickets[tickets["text"].map(lambda s: est_tokens(s) > MAX_INPUT_TOKENS)]
    print(f"empty tickets: {empty} | long tickets: {len(long_)} (estimated tokens: {sorted(set(long_['text'].map(est_tokens)))})")
    counts = {"email": 0, "phone": 0, "order_id": 0}
    redacted = tickets["text"].map(lambda s: redact(s, counts))
    print("substitutions:", counts, "| total:", sum(counts.values()))
    print("PII patterns still found in redacted texts:", int(redacted.map(has_pii).sum()))
    print("est_tokens zero-shot prompt:", est_tokens(ZERO_SHOT), "| few-shot prompt:", est_tokens(FEW_SHOT))

    # --- full runs, no budget
    zdf, zstats, zlog, zsent = run_pipeline(tickets, ZERO_SHOT)
    classified, truth = summarise("zero-shot, no budget", zdf, zstats, zlog)
    print("retry log (first 3 lines):", *zlog[:3], sep="\n  ")
    print("all prompts PII-free:", all(not has_pii(m["content"]) for msgs in zsent for m in msgs))
    print("confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(pd.Series(truth, name="true"), pd.Series(classified["category"].values, name="predicted")).to_string())
    errs = pd.Series(list(zip(truth, classified["category"].values)))
    print("errors:", errs[[a != b for a, b in errs]].value_counts().to_dict())

    fdf, fstats, flog, fsent = run_pipeline(tickets, FEW_SHOT)
    summarise("few-shot, no budget", fdf, fstats, flog)

    # --- budget demo (zero-shot, TOKEN_BUDGET = 6000)
    bdf, bstats, blog, _ = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
    summarise("zero-shot, TOKEN_BUDGET=6000", bdf, bstats, blog)
    print("budget_exhausted tickets:", int((bdf["status"] == "budget_exhausted").sum()))
    route(bdf, seg).to_csv("outputs/triage_budget_demo.csv", index=False)

    # --- final run (few-shot, TOKEN_BUDGET = 100000) with routing
    rdf, rstats, rlog, _ = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    summarise("final: few-shot, TOKEN_BUDGET=100000", rdf, rstats, rlog)
    final = route(rdf, seg)
    final.to_csv("outputs/triage_results.csv", index=False)
    print("\nrouting counts:", final["route"].value_counts().to_dict())
    print("priority counts:", final["priority"].value_counts().to_dict())
    print("rows in triage_results.csv:", len(final))
    with open("outputs/triage_retry_log.txt", "w") as f:
        f.write("\n".join(rlog) + "\n")
