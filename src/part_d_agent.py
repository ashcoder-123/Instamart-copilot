"""Part D (tasks 5-8): tools, ReAct loop, safe stops, checkpoint/resume, scenario runs and demos.
Run from the repository root:  python src/part_d_agent.py"""
import datetime
import glob
import json
import os
import sqlite3
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import mock_llm  # noqa: E402
from part_d_rag import retrieve  # noqa: E402

DB, LEDGER, RUNS, STATE = "data/instamart.db", "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "state"
CURRENT = {"run_id": None}          # lets the ledger record which run issued an action

AGENT_SYSTEM = ("You are an AGENT that handles customer refund requests for Instamart. Use the tools get_order, "
                "search_policy, calculate_refund, issue_refund and escalate_to_human. When finished, reply with one "
                "line starting with REFUND:, ESCALATE:, DENIED: or NOT_FOUND:.")


class SimulatedCrash(Exception):
    """Raised on purpose to demonstrate resuming from a checkpoint."""


# ------------------------------------------------------------------ 5. tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, minutes = row
    return {"order_id": oid, "status": status, "order_amount": float(amount),
            "delivery_minutes": None if minutes is None else float(minutes)}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    line = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tool": tool,
            "run_id": CURRENT["run_id"], "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by id in the Instamart database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the three best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


def execute_tool(name, arguments):
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**arguments)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(st, final=None):
    os.makedirs(STATE, exist_ok=True)
    data = {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "messages": st["messages"],
            "tokens_used": st["tokens_used"], "step": len(st["steps"]), "tools_executed": st["tools_executed"],
            "llm_calls": st["llm_calls"], "steps": st["steps"], "params": st["params"],
            "user_message": st["user_message"], "note": st["note"], "final": final}
    with open(os.path.join(STATE, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _refund_amount(steps):
    for s in reversed(steps):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"].get("refund_inr", 0.0)
    return 0.0


def _finish(st, status, final_answer):
    result = {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final_answer,
              "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"],
              "refund_inr": _refund_amount(st["steps"]), "tools_executed": st["tools_executed"]}
    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({**{k: result[k] for k in ("run_id", "scenario_id", "status", "final_answer", "llm_calls",
                                                       "tokens_used", "refund_inr")},
                            "note": st["note"], "steps": st["steps"]}) + "\n")
    save_checkpoint(st, final={k: v for k, v in result.items()})
    return result


def _fallback(st, status, reason):
    done = ", ".join(st["tools_executed"]) or "none"
    summary = (f"HANDOFF to human agent: {reason}. Scenario {st['scenario_id']}, {len(st['steps'])} steps completed "
               f"(tools run: {done}); no automatic decision was made, please review the request.")
    return _finish(st, status, summary)


# ------------------------------------------------------------------ 6/7. ReAct loop with safe stops
def _loop(st):
    p = st["params"]
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= p["time_limit_seconds"]:                      # (c) wall-clock limit
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {p['time_limit_seconds']} s reached")
        if st["llm_calls"] >= p["max_iterations"]:                                 # (a) iteration limit
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {p['max_iterations']} model calls")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > p["token_limit"]:                                   # (b) token limit, tool not executed
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit of {p['token_limit']} exceeded ({st['tokens_used']} used)")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _finish(st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        args = dict(tc.arguments)
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        CURRENT["run_id"] = st["run_id"]
        observation = execute_tool(tc.name, args)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(observation)})
        st["steps"].append({"tool": tc.name, "arguments": args, "observation": observation})
        st["tools_executed"].append(tc.name)
        save_checkpoint(st)                                                         # after every observation
        if st["params"].get("crash_after_tools") == len(st["tools_executed"]):
            raise SimulatedCrash(f"simulated crash after tool #{len(st['tools_executed'])} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None, note=""):
    st = {"run_id": run_id or f"{scenario_id}_{int(time.time())}", "scenario_id": scenario_id,
          "user_message": user_message, "note": note,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
          "params": {"max_iterations": max_iterations, "token_limit": token_limit,
                     "time_limit_seconds": time_limit_seconds, "crash_after_tools": crash_after_tools}}
    return _loop(st)


def resume_agent(run_id):
    """Load state/<run_id>.json and continue from the saved messages without repeating executed tools."""
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        c = json.load(f)
    if c.get("final"):
        return c["final"]
    params = dict(c["params"]); params["crash_after_tools"] = None
    st = {"run_id": c["run_id"], "scenario_id": c["scenario_id"], "user_message": c["user_message"], "note": c["note"],
          "messages": c["messages"], "tokens_used": c["tokens_used"], "llm_calls": c["llm_calls"],
          "steps": c["steps"], "tools_executed": c["tools_executed"], "params": params}
    return _loop(st)


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    for path in glob.glob(os.path.join(STATE, "*.json")) + [LEDGER, RUNS]:
        if os.path.exists(path):
            os.remove(path)
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    for r in sc.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id, run_id=r.scenario_id)
        expected = r.expected_outcome
        ok = res["status"].startswith("FALLBACK") if expected == "FALLBACK" else res["status"] == expected
        ok = ok and abs(res["refund_inr"] - r.expected_refund_inr) < 0.01
        rows.append({"scenario": r.scenario_id, "status": res["status"], "refund_inr": res["refund_inr"],
                     "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "expected": expected,
                     "match": ok})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    assert tab["match"].all(), "a scenario did not match its expectation"

    s1 = sc.iloc[0]
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token200", note="demo: token_limit=200")
    print(f"\nS1 with token_limit=200 -> {d1['status']} after {d1['llm_calls']} model calls, {d1['tokens_used']} tokens")
    print("  final_answer:", d1["final_answer"])
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout0", note="demo: time_limit_seconds=0")
    print(f"S1 with time_limit_seconds=0 -> {d2['status']} after {d2['llm_calls']} model calls")
    print("  final_answer:", d2["final_answer"])

    try:
        run_agent(s1.message, "S1", run_id="S1_crash", crash_after_tools=2, note="demo: crash after 2nd tool, then resume")
    except SimulatedCrash as e:
        print("\ncrash:", e)
    ck = json.load(open(os.path.join(STATE, "S1_crash.json")))
    print("tools executed before the crash:", ck["tools_executed"])
    res = resume_agent("S1_crash")
    print("tools executed across crash and resume:", res["tools_executed"])
    print(f"resumed run final: {res['status']} {res['refund_inr']:.2f} | model calls {res['llm_calls']} | tokens {res['tokens_used']}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) < 0.01
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| agent_runs lines:", sum(1 for _ in open(RUNS)),
          "| state files:", len(glob.glob(os.path.join(STATE, "*.json"))))
