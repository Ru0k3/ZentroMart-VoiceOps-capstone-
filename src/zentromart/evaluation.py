"""Practice-set evaluation with privacy-conscious, auditable JSONL output.

Fact matching is deliberately literal after Unicode/punctuation normalization.
It is a useful regression signal, not a semantic quality metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .agent import AgentSession, ZentroMartAgent
from .dataset import find_dataset_root
from .factory import create_agent


def _parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def _includes(output: str, required: str) -> bool:
    alternatives = [part.strip() for part in required.split("/") if part.strip()]
    normalized = _normalize(output)
    return any(_normalize(part) in normalized for part in alternatives)


def _fact_metrics(answer: str, must_include: str, must_not_include: str) -> dict[str, Any]:
    required = _parts(must_include)
    forbidden = _parts(must_not_include)
    missing = [item for item in required if not _includes(answer, item)]
    present_forbidden = [item for item in forbidden if _includes(answer, item)]
    return {
        "required_count": len(required),
        "required_matched": len(required) - len(missing),
        "missing_required": missing,
        "forbidden_found": present_forbidden,
        "required_pass": not missing,
        "forbidden_pass": not present_forbidden,
    }


def _evidence_metrics(cited: Iterable[str], gold: str) -> dict[str, Any]:
    expected = _parts(gold)
    actual = set(cited)
    matched = [item for item in expected if item in actual]
    return {
        "gold_count": len(expected),
        "matched": matched,
        "recall": (len(matched) / len(expected)) if expected else None,
    }


def _tool_expectations(value: str) -> list[dict[str, str]]:
    expected: list[dict[str, str]] = []
    for entry in _parts(value):
        match = re.fullmatch(r"(lookup_order|create_ticket)\s*\((.*?)\)", entry)
        if not match:
            continue
        name, arguments = match.groups()
        order = re.search(r"order_id\s*=\s*['\"]?(\d+)['\"]?", arguments)
        proof = re.search(r"verification_value\s*=\s*['\"]?(\d+)['\"]?", arguments)
        expected.append({
            "name": name,
            **({"order_id": order.group(1)} if order else {}),
            **({"verification_supplied": "true"} if proof else {}),
        })
    return expected


def _tool_metrics(expected_value: str, actual: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _tool_expectations(expected_value)
    remaining = list(actual)
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for item in expected:
        found = next((event for event in remaining if (
            event.get("name") == item["name"]
            and ("order_id" not in item or re.sub(r"\D", "", str(event.get("order_id", ""))) == item["order_id"])
            and ("verification_supplied" not in item or bool(event.get("verification_supplied")))
        )), None)
        if found is None:
            missing.append(item)
        else:
            remaining.remove(found)
            matched.append({"name": item["name"], "ok": bool(found.get("verified", found.get("created", True)))})
    unexpected = [event.get("name", "unknown") for event in remaining]
    return {
        "expected": expected,
        "matched": matched,
        "missing": missing,
        "unexpected": unexpected,
        "pass": not missing and not unexpected,
    }


def _case_record(
    *, case_id: str, mode: str, turn: Any, expected_decisions: list[str],
    must_include: str, must_not_include: str, gold_evidence: str,
    expected_tools: str = "",
) -> dict[str, Any]:
    answer = turn.answer
    decision = turn.decision
    fact = _fact_metrics(answer, must_include, must_not_include)
    evidence = _evidence_metrics(turn.cited_evidence_ids, gold_evidence)
    decision_pass = decision in expected_decisions if decision is not None else False
    tool = _tool_metrics(expected_tools, turn.tool_calls)
    strict_pass = decision_pass and fact["required_pass"] and fact["forbidden_pass"] and tool["pass"]
    if evidence["gold_count"]:
        strict_pass = strict_pass and evidence["recall"] == 1.0
    return {
        "case_id": case_id,
        "mode": mode,
        "answer": answer,
        "decision": decision,
        "expected_decisions": expected_decisions,
        "decision_pass": decision_pass,
        "facts": fact,
        "evidence": evidence,
        "tool_calls": tool,
        "strict_pass": strict_pass,
        "retrieved_evidence": turn.retrieved_evidence,
        "cited_evidence_ids": turn.cited_evidence_ids,
        "usage": turn.usage,
    }


def evaluate_questions(
    agent: ZentroMartAgent,
    path: Path,
    limit: int | None,
    *,
    offset: int = 0,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    results: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["asr_text"].strip() or row["question"].strip()
        expected = _parts(row.get("acceptable_decisions", "")) or [row.get("expected_decision", "")]
        turn = agent.respond(prompt, agent.new_session())
        record = _case_record(
            case_id=row["id"], mode=agent.mode, turn=turn,
            expected_decisions=expected,
            must_include=row.get("must_include", ""),
            must_not_include=row.get("must_not_include", ""),
            gold_evidence=row.get("gold_evidence", ""),
        )
        results.append(record)
        if on_record:
            on_record(record)
    return results


def evaluate_conversations(
    agent: ZentroMartAgent,
    path: Path,
    limit: int | None,
    *,
    offset: int = 0,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["conversation_id"]].append(row)
    selected = list(grouped.items())[offset:]
    if limit is not None:
        selected = selected[:limit]
    results: list[dict[str, Any]] = []
    for conversation_id, turns in selected:
        session: AgentSession = agent.new_session()
        for row in turns:
            turn = agent.respond(row["customer_message"], session)
            record = _case_record(
                case_id=f"{conversation_id}:turn-{row['turn']}",
                mode=agent.mode,
                turn=turn,
                expected_decisions=[row.get("expected_decision", "")],
                must_include=row.get("must_include", ""),
                must_not_include=row.get("must_not_include", ""),
                gold_evidence=row.get("gold_evidence", ""),
                expected_tools=row.get("expected_tool_calls", ""),
            )
            results.append(record)
            if on_record:
                on_record(record)
    return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(records),
        "strict_pass": sum(bool(record["strict_pass"]) for record in records),
        "decision_pass": sum(bool(record["decision_pass"]) for record in records),
        "required_fact_pass": sum(bool(record["facts"]["required_pass"]) for record in records),
        "forbidden_fact_pass": sum(bool(record["facts"]["forbidden_pass"]) for record in records),
        "tool_expectation_pass": sum(bool(record["tool_calls"]["pass"]) for record in records),
        "evidence_gold_items": 0,
        "evidence_matched_items": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    for record in records:
        summary["evidence_gold_items"] += record["evidence"]["gold_count"]
        summary["evidence_matched_items"] += len(record["evidence"]["matched"])
        for key in summary["usage"]:
            summary["usage"][key] += int(record["usage"].get(key, 0))
    gold = summary["evidence_gold_items"]
    summary["evidence_recall"] = summary["evidence_matched_items"] / gold if gold else None
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ZentroMart practice cases.")
    parser.add_argument("--mode", choices=("baseline", "rag", "conversation", "all"), default="rag")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset directory (defaults to ZENTROMART_DATASET_DIR or bundled local path).")
    parser.add_argument("--limit", type=int, default=1, help="Question rows and conversation groups per mode (default: 1).")
    parser.add_argument("--all-cases", action="store_true", help="Run the full practice set; may use substantial NVIDIA credits.")
    parser.add_argument("--question-offset", type=int, default=0, help="Skip this many practice questions (for checkpointed batches).")
    parser.add_argument("--conversation-offset", type=int, default=0, help="Skip this many conversation groups (for checkpointed batches).")
    parser.add_argument("--retry-rate-limits", type=int, default=0, help="Explicitly retry NVIDIA HTTP 429 responses this many times; disabled by default.")
    parser.add_argument("--rate-limit-retry-delay", type=float, default=60.0, help="Wait this many seconds when NVIDIA gives no Retry-After header (default: 60).")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.jsonl"))
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.question_offset < 0:
        parser.error("--question-offset cannot be negative")
    if args.conversation_offset < 0:
        parser.error("--conversation-offset cannot be negative")
    if args.retry_rate_limits < 0:
        parser.error("--retry-rate-limits cannot be negative")
    if args.rate_limit_retry_delay < 0:
        parser.error("--rate-limit-retry-delay cannot be negative")
    limit = None if args.all_cases else args.limit
    root = find_dataset_root(args.dataset)
    questions = root / "practice" / "questions.csv"
    conversations = root / "practice" / "conversations.csv"
    modes = ("baseline", "rag", "conversation") if args.mode == "all" else (args.mode,)
    question_modes = sum(mode in {"baseline", "rag"} for mode in modes)
    with questions.open(encoding="utf-8-sig", newline="") as handle:
        question_count = sum(1 for _ in csv.DictReader(handle))
    question_count = max(0, question_count - args.question_offset)
    question_count = min(question_count, limit) if limit is not None else question_count
    estimated = question_count * question_modes
    # The agent permits one support action followed by one final response.
    if "conversation" in modes:
        selected_groups = list(_read_groups(conversations).values())[args.conversation_offset:]
        if limit is not None:
            selected_groups = selected_groups[:limit]
        estimated += sum(len(group) for group in selected_groups) * 2
    print(f"Estimated model-call upper bound: {estimated}; full-set runs can consume credits.", file=sys.stderr)

    all_records: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        def checkpoint(record: dict[str, Any]) -> None:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        for mode in modes:
            agent = create_agent(
                mode,
                dataset_root=root,
                rate_limit_retries=args.retry_rate_limits,
                rate_limit_retry_delay=args.rate_limit_retry_delay,
            )
            if mode == "conversation":
                records = evaluate_conversations(
                    agent,
                    conversations,
                    limit,
                    offset=args.conversation_offset,
                    on_record=checkpoint,
                )
            else:
                records = evaluate_questions(
                    agent,
                    questions,
                    limit,
                    offset=args.question_offset,
                    on_record=checkpoint,
                )
            all_records.extend(records)
    summary = summarize(all_records)
    summary["by_mode"] = {
        mode: summarize([record for record in all_records if record["mode"] == mode])
        for mode in modes
    }
    print(json.dumps({"output": str(args.output), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


def _read_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["conversation_id"]].append(row)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
