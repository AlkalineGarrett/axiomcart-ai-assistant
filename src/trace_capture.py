"""
Trace capture — run the real graph and record every prompt, model reply,
tool call and interrupt to a JSON file.

    uv run python -m src.trace_capture
        one canned turn: the README query, answering ORD102 when asked

    uv run python -m src.trace_capture --interactive
        a REPL — you type the queries and answer the agent yourself.
        Every turn lands in the same trace; 'quit' writes the file.

    uv run python -m src.trace_capture --query "where is ORD101?"
    uv run python -m src.trace_capture --answer ORD102 --answer "yes please"
    uv run python -m src.trace_capture --query "hi" --interactive -o my-run.json
        run that query first, then keep going in the REPL

Nothing under src/ is modified. This module patches module-level globals
in src.nodes — product_llm, sales_llm, llm, the tool registries and
interrupt — and wraps the four node functions *before* src.graph compiles
the graph, so what gets recorded is the real run, not a simulation.

Two details worth knowing when you read the output:

  • Each HITL question records two interrupt events. Resuming re-runs
    the ask_user node from the top, so interrupt() is called again and
    this time returns the answer. The first event is marked "raised": true.
    (Traces captured before ask_user existed also show the support model
    call twice — it shared a node with interrupt() and re-ran on resume.)
  • Structured-output calls (the orchestrator's classifier) report no
    token usage — LangChain hands back the parsed Pydantic object and
    drops the raw response that carries usage_metadata.

Output: artifacts/trace.json, loadable by the Prompt Trace tab of
graph-walkthrough.html.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone

# ── recording state ──────────────────────────────────────────

_events: list[dict] = []
_lock = threading.Lock()
_t0 = time.perf_counter()
_llm_n = 0
_node = ContextVar("axiomcart_node", default=None)


def _rec(ev: dict) -> dict:
    """Append an event as it *starts*, so ordering reflects the real sequence."""
    with _lock:
        ev["seq"] = len(_events) + 1
        ev["t"] = round(time.perf_counter() - _t0, 3)
        _events.append(ev)
    return ev


def _next_llm_n() -> int:
    global _llm_n
    with _lock:
        _llm_n += 1
        return _llm_n


# ── serialisation ────────────────────────────────────────────

def _text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


def _ser_msg(m) -> dict:
    """One LangChain message → a plain dict."""
    if isinstance(m, str):
        return {"role": "human", "text": m}
    d = {"role": getattr(m, "type", type(m).__name__), "text": _text(getattr(m, "content", ""))}
    tcs = getattr(m, "tool_calls", None) or []
    if tcs:
        d["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args")} for tc in tcs]
    if getattr(m, "tool_call_id", None):
        d["tool_call_id"] = m.tool_call_id
    return d


def _ser_sent(x) -> list[dict]:
    return [_ser_msg(m) for m in x] if isinstance(x, (list, tuple)) else [_ser_msg(x)]


def _ser_reply(r, structured: bool = False) -> dict:
    """An AIMessage (or a parsed Pydantic object) → a plain dict.

    Note: AIMessage is itself a Pydantic model in langchain-core 1.x, so
    "has model_dump()" does NOT mean structured output. The caller knows
    which kind of runnable it wrapped; trust that, and fall back to duck
    typing (a message has both .type and .content).
    """
    is_message = hasattr(r, "type") and hasattr(r, "content")
    if (structured or not is_message) and hasattr(r, "model_dump"):
        return {"structured": json.loads(json.dumps(r.model_dump(), default=str))}

    out = {"content": _text(getattr(r, "content", r))}
    tcs = getattr(r, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args")} for tc in tcs]
    usage = getattr(r, "usage_metadata", None) or {}
    if usage:
        out["tokens"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")}
    meta = getattr(r, "response_metadata", None) or {}
    for src_key, dst in (("model_name", "model"), ("finish_reason", "finish_reason")):
        if meta.get(src_key):
            out[dst] = meta[src_key]
    return out


# ── proxies ──────────────────────────────────────────────────

class _TracedLLM:
    """Wraps a chat model (or a bound/structured runnable) and records invoke()."""

    def __init__(self, inner, label: str, kind: str = "chat"):
        self._inner = inner
        self._label = label
        self._kind = kind

    def invoke(self, input, config=None, **kw):
        ev = _rec({
            "k": "llm",
            "n": _next_llm_n(),
            "node": _node.get() or self._label,
            "label": self._label,
            "structured": self._kind == "structured",
            "sent": _ser_sent(input),
        })
        started = time.perf_counter()
        try:
            reply = self._inner.invoke(input, config, **kw)
        except BaseException as exc:
            ev["secs"] = round(time.perf_counter() - started, 3)
            ev["error"] = f"{type(exc).__name__}: {exc}"
            raise
        ev["secs"] = round(time.perf_counter() - started, 3)
        ev["out"] = _ser_reply(reply, structured=self._kind == "structured")
        return reply

    # keep the chain traced when nodes.py derives new runnables
    def with_structured_output(self, *a, **kw):
        return _TracedLLM(self._inner.with_structured_output(*a, **kw), self._label, "structured")

    def bind_tools(self, *a, **kw):
        return _TracedLLM(self._inner.bind_tools(*a, **kw), self._label, self._kind)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TracedTool:
    """Wraps a @tool and records its arguments and output."""

    def __init__(self, inner):
        self._inner = inner

    def invoke(self, args, config=None, **kw):
        ev = _rec({
            "k": "tool",
            "name": getattr(self._inner, "name", "?"),
            "node": _node.get(),
            "args": json.loads(json.dumps(args, default=str)),
        })
        started = time.perf_counter()
        try:
            out = self._inner.invoke(args, config, **kw)
        except BaseException as exc:
            ev["secs"] = round(time.perf_counter() - started, 3)
            ev["error"] = f"{type(exc).__name__}: {exc}"
            raise
        ev["secs"] = round(time.perf_counter() - started, 3)
        ev["out"] = _text(out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TracedGraph:
    """Wraps the compiled graph to mark invoke() boundaries."""

    def __init__(self, inner):
        self._inner = inner

    def invoke(self, input, config=None, **kw):
        resume = getattr(input, "resume", None)
        ev = _rec({
            "k": "invoke",
            "kind": "resume" if resume is not None else "start",
            "input": {"resume": _text(resume)} if resume is not None else _ser_state(input),
        })
        started = time.perf_counter()
        out = self._inner.invoke(input, config, **kw)
        ev["secs"] = round(time.perf_counter() - started, 3)
        ev["interrupted"] = bool(isinstance(out, dict) and out.get("__interrupt__"))
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _ser_state(state) -> dict:
    if not isinstance(state, dict):
        return {"input": _text(state)}
    out = {}
    if "user_query" in state:
        out["user_query"] = state["user_query"]
    if state.get("messages"):
        out["messages"] = [_ser_msg(m) for m in state["messages"]]
    return out


# ── installation ─────────────────────────────────────────────

def _is_interrupt(exc: BaseException) -> bool:
    try:
        from langgraph.errors import GraphInterrupt
        return isinstance(exc, GraphInterrupt)
    except Exception:
        return "Interrupt" in type(exc).__name__


def _wrap_node(fn, name: str):
    @functools.wraps(fn)
    def wrapper(state, *a, **kw):
        _rec({"k": "node", "name": name, "phase": "enter"})
        token = _node.set(name)
        started = time.perf_counter()
        try:
            out = fn(state, *a, **kw)
        except BaseException as exc:
            _rec({"k": "node", "name": name, "phase": "exit",
                  "outcome": "interrupted" if _is_interrupt(exc) else type(exc).__name__,
                  "secs": round(time.perf_counter() - started, 3)})
            raise
        finally:
            _node.reset(token)
        _rec({"k": "node", "name": name, "phase": "exit", "outcome": "ok",
              "secs": round(time.perf_counter() - started, 3)})
        return out

    return wrapper


def _install(nodes) -> None:
    """Patch src.nodes in place. Must run before src.graph is imported."""

    # Model calls. These are module globals, looked up at call time, so the
    # already-compiled agent subgraphs pick the wrappers up too.
    nodes.llm = _TracedLLM(nodes.llm, "llm")
    nodes.product_llm = _TracedLLM(nodes.product_llm, "product_agent")
    nodes.sales_llm = _TracedLLM(nodes.sales_llm, "support_agent")

    # Tool execution.
    nodes.product_tools_by_name = {k: _TracedTool(v) for k, v in nodes.product_tools_by_name.items()}
    nodes.sales_tools_by_name = {k: _TracedTool(v) for k, v in nodes.sales_tools_by_name.items()}

    # HITL. On the first pass interrupt() raises; on resume it returns the answer.
    real_interrupt = nodes.interrupt

    def traced_interrupt(value):
        ev = _rec({"k": "interrupt", "node": _node.get(), "question": _text(value)})
        try:
            answer = real_interrupt(value)
        except BaseException:
            ev["raised"] = True          # suspended the graph; no answer on this pass
            raise
        ev["answer"] = _text(answer)
        return answer

    nodes.interrupt = traced_interrupt

    # Node boundaries — patched before graph.py imports these names.
    for name in ("orchestrator_node", "product_agent", "support_agent", "synthesizer_node"):
        setattr(nodes, name, _wrap_node(getattr(nodes, name), name))


# ── run ──────────────────────────────────────────────────────

DEFAULT_QUERY = "My order is delayed and I want to see some headphones"


def capture(queries: list[str], answers: list[str], interactive: bool) -> dict:
    """Run `queries`, then hand over to a REPL if `interactive`.

    One assistant instance means one thread_id, so every turn shares the
    conversation history — exactly like a real session.
    """
    global _t0

    import src.nodes as nodes                    # imports config + builds the vector store
    _install(nodes)

    from src.graph import axiomcart_graph        # compiles with the wrapped nodes
    import src.main as app
    from src.config import llm as base_llm

    app.axiomcart_graph = _TracedGraph(axiomcart_graph)
    assistant = app.AxiomCartAssistant(enable_voice=False)

    pending = list(answers)
    used: list[str] = []

    def input_fn(question: str) -> str:
        if pending:                                    # scripted --answer, in order
            reply = pending.pop(0)
            print(f"\n🔄 Agent asks: {question}\nYou: {reply}   [scripted]")
        elif interactive:                              # you answer it
            reply = input(f"\n🔄 Agent asks: {question}\nYou: ").strip()
        else:                                          # nothing left and nobody to ask
            reply = "-"
            print(f"\n🔄 Agent asks: {question}\nYou: (no --answer left, replying '-')")
        used.append(reply)
        return reply

    turns: list[dict] = []

    def run_turn(text: str) -> str:
        n = len(turns) + 1
        _rec({"k": "turn", "n": n, "query": text})
        started = time.perf_counter()
        reply = assistant.query(text, input_fn=input_fn)
        _rec({"k": "final", "turn": n, "text": reply,
              "secs": round(time.perf_counter() - started, 3)})
        turns.append({"n": n, "query": text, "answer": reply})
        return reply

    _events.clear()
    _t0 = time.perf_counter()
    started = time.perf_counter()

    for q in queries:
        print(f"\n▶  {q}\n")
        print(f"\nAssistant: {run_turn(q)}\n")

    if interactive:
        print("\n🛒  capture REPL — type a query, 'quit' to stop and write the trace\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit", "bye"):
                break
            print(f"\nAssistant: {run_turn(text)}\n")

    elapsed = round(time.perf_counter() - started, 2)

    if not turns:
        sys.exit("nothing captured — no query was run")

    llm_events = [e for e in _events if e["k"] == "llm"]
    tok_in = sum((e.get("out", {}).get("tokens") or {}).get("input_tokens") or 0 for e in llm_events)
    tok_out = sum((e.get("out", {}).get("tokens") or {}).get("output_tokens") or 0 for e in llm_events)

    return {
        "meta": {
            "query": turns[0]["query"],
            "answer": turns[-1]["answer"],
            "turns": len(turns),
            "queries": [t["query"] for t in turns],
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": getattr(base_llm, "model_name", None) or getattr(base_llm, "model", "?"),
            "temperature": getattr(base_llm, "temperature", None),
            "hitl_answers": used,
            "llm_calls": len(llm_events),
            "tool_calls": len([e for e in _events if e["k"] == "tool"]),
            "invocations": len([e for e in _events if e["k"] == "invoke"]),
            "interrupts": len([e for e in _events if e["k"] == "interrupt"]),
            "duration_s": elapsed,
            "tokens": {"input": tok_in, "output": tok_out},
        },
        "events": list(_events),
    }


def summarise(payload: dict) -> None:
    m = payload["meta"]
    print("\n" + "─" * 78)
    print(f"{'seq':>4}  {'t':>6}  {'secs':>6}  kind      detail")
    print("─" * 78)
    for e in payload["events"]:
        k = e["k"]
        if k == "llm":
            detail = f"#{e['n']} {e['node']}  →  " + (
                ", ".join(tc["name"] + "(…)" for tc in e.get("out", {}).get("tool_calls", []))
                or ("structured output" if e.get("structured") else f"{len(e.get('out', {}).get('content', ''))} chars")
            )
        elif k == "tool":
            detail = f"{e['name']}({json.dumps(e.get('args', {}))})  →  {len(e.get('out', ''))} chars"
        elif k == "node":
            detail = f"{e['name']} {e['phase']}" + (f" ({e['outcome']})" if e.get("outcome") else "")
        elif k == "interrupt":
            detail = "raised — graph suspended" if e.get("raised") else f"resumed with {e.get('answer')!r}"
        elif k == "invoke":
            detail = e["kind"] + (" → interrupted" if e.get("interrupted") else "")
        elif k == "turn":
            detail = f"turn {e['n']}: {e['query']!r}"
        else:
            detail = f"{len(e.get('text', ''))} chars"
        print(f"{e['seq']:>4}  {e['t']:>6}  {e.get('secs', ''):>6}  {k:<8}  {detail[:60]}")
    print("─" * 78)
    print(f"{m.get('turns', 1)} turn(s) · {m['llm_calls']} LLM calls · {m['tool_calls']} tool calls · "
          f"{m['interrupts']} interrupts · {m['invocations']} invoke() · {m['duration_s']}s · "
          f"{m['tokens']['input']}+{m['tokens']['output']} tokens (structured calls report none)")


def main() -> None:
    p = argparse.ArgumentParser(description="Run the graph and capture a full prompt/response trace")
    p.add_argument("--query", default=None,
                   help=f"query to send (default: {DEFAULT_QUERY!r}; with --interactive, "
                        "runs this first and then hands over to the REPL)")
    p.add_argument("--answer", action="append", default=[],
                   help="scripted reply for a HITL interrupt (repeatable, in order)")
    p.add_argument("--interactive", action="store_true",
                   help="drive the session yourself: type each query and answer the agent's "
                        "questions. Every turn is captured; 'quit' writes the trace")
    p.add_argument("-o", "--out", default="artifacts/trace.json", help="output file (default: artifacts/trace.json)")
    args = p.parse_args()

    # --interactive means you drive: nothing canned unless you asked for it with --query.
    if args.query:
        queries = [args.query]
    elif args.interactive:
        queries = []
    else:
        queries = [DEFAULT_QUERY]

    answers = args.answer or ([] if args.interactive else ["ORD102"])

    try:
        payload = capture(queries, answers, args.interactive)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)

    summarise(payload)
    print(f"\nwrote {args.out} — load it in the Prompt Trace tab of graph-walkthrough.html")


if __name__ == "__main__":
    main()
