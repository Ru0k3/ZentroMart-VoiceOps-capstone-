"""Prompt-only, RAG, and tool-using ZentroMart agent stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Literal

from .dataset import Dataset
from .knowledge import KnowledgeRetriever, SearchResult
from .llm import NVIDIAChatClient
from .prompts import BASELINE_SYSTEM_PROMPT, CONVERSATION_RULES, RAG_SYSTEM_PROMPT
from .tools import ConversationToolGuard


AgentMode = Literal["baseline", "rag", "conversation"]
_SOURCE_TAG = re.compile(r"\s*\[SOURCES:\s*(.*?)\]\s*$", re.IGNORECASE | re.DOTALL)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an order mentioned by the caller. Call without "
                "verification_value first to learn the required verification "
                "step. Only pass a phone suffix or pincode the caller just "
                "provided; never guess it. Details are returned only after a match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "verification_value": {"type": "string"},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a human handover ticket for a support issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "intent": {"type": "string"},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class AgentSession:
    """In-memory context for one conversation; never written to disk."""

    tool_guard: ConversationToolGuard | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    ticket_created: bool = False


@dataclass(frozen=True)
class AgentTurn:
    answer: str
    mode: AgentMode
    retrieved_evidence: list[dict[str, str]]
    cited_evidence_ids: list[str]
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int]


class ZentroMartAgent:
    """Run a staged support agent on NVIDIA Build chat completions."""

    def __init__(
        self,
        client: NVIDIAChatClient,
        dataset: Dataset,
        retriever: KnowledgeRetriever,
        *,
        mode: AgentMode = "conversation",
        top_k: int = 5,
        max_tool_rounds: int = 3,
    ):
        if mode not in {"baseline", "rag", "conversation"}:
            raise ValueError(f"Unsupported agent mode: {mode}")
        self.client = client
        self.dataset = dataset
        self.retriever = retriever
        self.mode = mode
        self.top_k = top_k
        self.max_tool_rounds = max_tool_rounds

    def new_session(self) -> AgentSession:
        guard = ConversationToolGuard(self.dataset) if self.mode == "conversation" else None
        return AgentSession(tool_guard=guard)

    def respond(self, message: str, session: AgentSession | None = None) -> AgentTurn:
        if not message.strip():
            raise ValueError("Customer message must not be empty")
        session = session or self.new_session()
        use_rag = self.mode != "baseline"
        use_tools = self.mode == "conversation"

        query = self._retrieval_query(session.messages, message)
        results = self.retriever.search(query, top_k=self.top_k) if use_rag else []
        system_prompt = self._system_prompt(results)
        base_history = list(session.messages)
        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *base_history,
            {"role": "user", "content": message},
        ]
        if session.tool_guard is not None:
            session.tool_guard.record_caller_message(message)

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_events: list[dict[str, Any]] = []
        answer = ""

        for round_index in range(self.max_tool_rounds + 1):
            completion = self.client.complete(
                request_messages,
                tools=TOOL_DEFINITIONS if use_tools else None,
                temperature=0.1,
                max_tokens=1024,
            )
            for key in usage:
                usage[key] += completion.usage.get(key, 0)
            assistant_message = completion.message
            calls = assistant_message.get("tool_calls") or []
            if not calls:
                answer = str(assistant_message.get("content") or "").strip()
                break
            if not use_tools or round_index >= self.max_tool_rounds:
                answer = (
                    "I couldn't complete that request safely. Please try again "
                    "or ask for a human support colleague."
                )
                break

            # Keep tool-call protocol messages only inside this request. The
            # session history saved below is compacted to redacted text turns.
            request_messages.append({
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": self._sanitize_tool_calls(calls),
                **({"reasoning_content": assistant_message["reasoning_content"]}
                   if assistant_message.get("reasoning_content") else {}),
            })
            for call in calls:
                if len(tool_events) >= 4:
                    result = {"error": "Per-turn support action limit reached."}
                else:
                    result, event = self._run_tool(call, session, message)
                    tool_events.append(event)
                    if event.get("name") == "create_ticket" and event.get("created"):
                        session.ticket_created = True
                request_messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                })

        if not answer:
            answer = "I'm sorry, I couldn't prepare a reliable answer just now."
        answer, cited_evidence_ids = self._extract_sources(answer, results)
        if session.tool_guard is not None:
            answer = session.tool_guard.redact(answer)
        if not session.ticket_created and re.search(
            r"\b(?:ticket (?:has been|was) created|"
            r"i(?:'ve| have)? created (?:a|the) ticket|"
            r"handover (?:has been|was) submitted|"
            r"i(?:'ve| have)? escalated|human colleague will contact you)\b",
            answer,
            flags=re.IGNORECASE,
        ):
            answer = (
                "I wasn't able to complete the handover yet, so I can't say a "
                "ticket was created. Please try again or contact a human support colleague."
            )

        # Keep only text turns for future context. This drops function-call
        # arguments, including verification values, from subsequent requests.
        session.messages = base_history + [
            {
                "role": "user",
                "content": (
                    session.tool_guard.redact(message)
                    if session.tool_guard is not None else message
                ),
            },
            {"role": "assistant", "content": answer},
        ]

        return AgentTurn(
            answer=answer,
            mode=self.mode,
            retrieved_evidence=[result.chunk.evidence() for result in results],
            cited_evidence_ids=cited_evidence_ids,
            tool_calls=tool_events,
            usage=usage,
        )

    def _system_prompt(self, results: list[SearchResult]) -> str:
        if self.mode == "baseline":
            return BASELINE_SYSTEM_PROMPT

        prompt = RAG_SYSTEM_PROMPT
        if self.mode == "conversation":
            prompt += "\n\n" + CONVERSATION_RULES
        if results:
            prompt += "\n\nRetrieved evidence for this turn:\n\n"
            prompt += "\n\n---\n\n".join(
                result.chunk.prompt_text() for result in results
            )
        else:
            prompt += (
                "\n\nNo policy or catalog evidence was retrieved for this turn. "
                "Do not invent company-specific facts."
            )
        prompt += (
            "\n\nA verified tool result in this conversation may support "
            "order-specific facts. Never treat an unverified lookup as evidence."
        )
        return prompt

    @staticmethod
    def _retrieval_query(history: list[dict[str, str]], message: str) -> str:
        recent_user_context = [
            item["content"] for item in history[-6:]
            if item.get("role") == "user" and item.get("content")
        ]
        return " ".join([*recent_user_context, message])

    @staticmethod
    def _extract_sources(
        answer: str,
        results: list[SearchResult],
    ) -> tuple[str, list[str]]:
        match = _SOURCE_TAG.search(answer)
        if not match:
            return answer.strip(), []
        answer = answer[:match.start()].strip()
        allowed = {result.chunk.section_id for result in results}
        raw_ids = [value.strip() for value in match.group(1).split(",")]
        cited = [section_id for section_id in raw_ids if section_id in allowed]
        return answer, cited

    @staticmethod
    def _sanitize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove caller verification proof before replaying tool calls to the LLM."""
        safe_calls = json.loads(json.dumps(calls))
        for call in safe_calls:
            function = call.get("function") or {}
            if function.get("name") != "lookup_order":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if "verification_value" in arguments:
                arguments["verification_value"] = "[verification redacted]"
                function["arguments"] = json.dumps(arguments)
        return safe_calls

    @staticmethod
    def _run_tool(
        call: dict[str, Any],
        session: AgentSession,
        caller_message: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        function = call.get("function") or {}
        name = function.get("name", "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
        except (json.JSONDecodeError, ValueError):
            return (
                {"error": "Invalid function arguments; ask the customer for clarification."},
                {"name": name or "unknown", "ok": False},
            )

        try:
            if session.tool_guard is None:
                return (
                    {"error": "Tools are not enabled for this agent stage."},
                    {"name": name or "unknown", "ok": False},
                )
            if name == "lookup_order":
                order_id = str(arguments.get("order_id", ""))
                proof = arguments.get("verification_value")
                result = session.tool_guard.lookup_order(
                    order_id,
                    caller_message=caller_message,
                    verification_value=str(proof) if proof is not None else None,
                )
                event = {
                    "name": name,
                    "order_id": order_id,
                    "verification_supplied": proof is not None,
                    "verified": bool(result.get("verified")),
                }
                return result, event
            if name == "create_ticket":
                result = session.tool_guard.create_ticket(**arguments)
                return result, {
                    "name": name,
                    "created": bool(result.get("created")),
                    "reason": arguments.get("reason", ""),
                }
        except (TypeError, ValueError, OSError):
            return (
                {"error": "The requested support action failed. Do not claim it succeeded."},
                {"name": name or "unknown", "ok": False},
            )

        return (
            {"error": "This function is not available."},
            {"name": name or "unknown", "ok": False},
        )
