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
_DECISION_TAG = re.compile(
    r"\s*\[DECISION:\s*(answer|clarify|escalate|refuse)\s*\]\s*$",
    re.IGNORECASE,
)
_CITATION_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "so", "the", "this", "to", "was", "we", "what", "when", "where",
    "which", "who", "with", "you", "your", "can", "could", "would",
    "should", "do", "does", "did", "have", "has", "had", "will", "am",
    "my", "our", "their", "them", "they", "there", "here", "order",
    "product", "item", "customer", "policy", "help", "zentromart", "sku",
    "है", "हैं", "हो", "का", "की", "के", "में", "से", "पर", "और",
    "क्या", "कैसे", "कहाँ", "मुझे", "मैं", "मेरा", "मेरी", "मेरे", "यह",
    "ये", "वह", "तो", "एक", "लिए", "कर", "करें", "रहा", "रही", "रहे",
    "कृपया", "बताइए", "आपका", "आपकी", "हूँ", "होगा", "होगी",
})

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
    decision: str | None
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
        max_tool_rounds: int = 1,
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
        lookup_failed = False

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
                if tool_events:
                    result = {
                        "error": (
                            "Only one support action is allowed per customer turn. "
                            "Continue with the result already returned."
                        )
                    }
                else:
                    result, event = self._run_tool(call, session, message)
                    tool_events.append(event)
                    if event.get("name") == "lookup_order" and not event.get("verified"):
                        lookup_failed = True
                    if event.get("name") == "create_ticket" and event.get("created"):
                        session.ticket_created = True
                request_messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                })

        if not answer:
            answer = "I'm sorry, I couldn't prepare a reliable answer just now."
        answer, cited_evidence_ids, decision = self._extract_markers(answer, results)
        answer = self._strip_citation_artifacts(answer, results)
        if decision is None:
            decision = self._infer_decision(answer)
        if self._must_request_order_verification(message, session, lookup_failed):
            answer = (
                "I can't share order-specific details until the order is verified. "
                "Please provide either the last four digits of the registered "
                "phone number or the delivery pincode."
            )
            cited_evidence_ids = []
            decision = "clarify"
        if session.tool_guard is not None:
            answer = session.tool_guard.redact(answer)
        if not session.ticket_created and self._claims_handover(answer):
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
            decision=decision,
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
    def _extract_markers(
        answer: str,
        results: list[SearchResult],
    ) -> tuple[str, list[str], str | None]:
        decision_match = _DECISION_TAG.search(answer)
        decision = decision_match.group(1).casefold() if decision_match else None
        if decision_match:
            answer = answer[:decision_match.start()] + answer[decision_match.end():]

        source_match = _SOURCE_TAG.search(answer)
        if not source_match:
            return (
                answer.strip(),
                ZentroMartAgent._overlap_citations(answer, results),
                decision,
            )
        source_text = source_match.group(1)
        answer = answer[:source_match.start()] + answer[source_match.end():]
        allowed = {result.chunk.section_id for result in results}
        raw_ids = [value.strip() for value in source_text.split(",")]
        cited = [section_id for section_id in raw_ids if section_id in allowed]
        return answer.strip(), cited, decision

    @staticmethod
    def _overlap_citations(
        answer: str,
        results: list[SearchResult],
    ) -> list[str]:
        """Cite only retrieved chunks with substantial informative overlap."""
        def terms(text: str) -> set[str]:
            return {
                token.casefold()
                for token in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
                if token.casefold() not in _CITATION_STOPWORDS
                and (len(token) > 2 or token.isdigit())
            }

        answer_terms = terms(answer)
        if not answer_terms:
            return []
        cited: list[str] = []
        for result in results:
            evidence_terms = terms(
                result.chunk.section_title + " " + result.chunk.content
            )
            overlap = answer_terms & evidence_terms
            cosine = len(overlap) / max(
                1.0, (len(answer_terms) * len(evidence_terms)) ** 0.5
            )
            evidence_coverage = len(overlap) / max(1, len(evidence_terms))
            if len(overlap) >= 3 and cosine >= 0.25 and evidence_coverage >= 0.15:
                cited.append(result.chunk.section_id)
        return cited

    @staticmethod
    def _strip_citation_artifacts(
        answer: str,
        results: list[SearchResult],
    ) -> str:
        """Keep source references in metadata rather than spoken customer text."""
        for result in results:
            evidence_id = result.chunk.section_id
            if ":" in evidence_id:
                answer = re.sub(
                    rf"\[?{re.escape(evidence_id)}\]?",
                    "",
                    answer,
                    flags=re.IGNORECASE,
                )
            else:
                answer = re.sub(
                    rf"\[{re.escape(evidence_id)}\]",
                    "",
                    answer,
                    flags=re.IGNORECASE,
                )
        answer = re.sub(
            r"\(?\s*(?:(?:source|section)\s+)?id\s*[:=]\s*"
            r"[A-Z]{2,}(?:-[A-Z0-9]+)+\s*\)?",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        return re.sub(r"[ \t]{2,}", " ", answer).strip()

    @staticmethod
    def _must_request_order_verification(
        message: str,
        session: AgentSession,
        lookup_failed: bool,
    ) -> bool:
        """Never trust model prose with order data until the guard verified it."""
        guard = session.tool_guard
        if guard is None:
            return False
        if lookup_failed:
            return True
        if not guard.unverified_order_ids:
            return False
        order_related = re.search(
            r"\b(order|delivery|delivered|shipment|shipped|tracking|refund|"
            r"replacement|parcel|item|product|status|price|amount)\b|"
            r"ऑर्डर|डिलीवरी|पार्सल|रिफंड|वापसी|मिला|मिली|भेजा|भेजी",
            message,
            flags=re.IGNORECASE,
        )
        return bool(order_related)

    @staticmethod
    def _claims_handover(answer: str) -> bool:
        answer = re.sub(
            r"(?:can't|cannot|could not|wasn't able to|not able to)\s+"
            r"(?:say|state|claim|confirm)\s+(?:that\s+)?"
            r".{0,50}?\bticket\b.{0,30}?\b(?:created|raised|submitted|opened)\b",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        return bool(re.search(
            r"(?:\bticket\b.{0,60}\b(?:created|raised|submitted|opened)\b|"
            r"\b(?:created|raised|submitted|opened)\b.{0,60}\bticket\b|"
            r"\b(?:i(?:'ve| have)? escalated|handover (?:has been|was) submitted|"
            r"human (?:support )?(?:agent|colleague) will contact you)\b|"
            r"टिकट.{0,60}(?:बन|बना|बनाया|बनाकर|दर्ज|भेज)|"
            r"(?:बन|बना|बनाया|बनाकर|दर्ज|भेज).{0,60}टिकट|"
            r"मानव सहायता.{0,50}(?:संपर्क|फोन|कॉल))",
            answer,
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _infer_decision(answer: str) -> str:
        """Fill internal decision metadata when a model omits its output tag."""
        opening = answer[:220]
        if ZentroMartAgent._claims_handover(answer) or re.search(
            r"\b(?:escalat(?:e|ed|ing)|human (?:support|agent|colleague)|"
            r"speak to (?:a )?person|connect you to)\b|"
            r"मानव सहायता|एजेंट से बात|जोड़ सकता",
            opening,
            flags=re.IGNORECASE,
        ):
            return "escalate"
        if re.search(
            r"\b(?:can't|cannot|unable to|won't|do not have access|"
            r"don't have access|not able to)\b.{0,100}\b(?:provide|share|help|confirm|"
            r"disclose|access|identify)\b|"
            r"नहीं बता सकता|नहीं बता सकती|साझा नहीं कर सकता|पुष्टि नहीं कर सकता",
            opening,
            flags=re.IGNORECASE,
        ):
            return "refuse"
        clarification = re.search(
            r"\b(?:could you (?:please )?(?:provide|share|confirm|tell)|"
            r"please (?:provide|share|tell me)|what is your|which would you|"
            r"do you have|may i have|i need (?:your|the)|need to know)\b|"
            r"(?:कृपया|बताइए).{0,50}(?:ऑर्डर|नंबर|पिन|बताएं|दीजिए)|"
            r"(?:क्या आप|कौन सा|कौन सी).{0,60}(?:बताएं|देंगे|देंगी|चाहते|चाहती)",
            answer,
            flags=re.IGNORECASE,
        )
        if clarification:
            prior_text = answer[:clarification.start()]
            substantive = re.search(
                r"\d|₹|\b(?:warranty|warranties|claim|refund|return|fee|"
                r"delivery|delivered|shipped|eligible|covered|waived|replacement|"
                r"product|item|status|price|amount)\b|"
                r"वारंटी|क्लेम|रिफ़ंड|रिटर्न|डिलीवरी|महीने|रुपये",
                prior_text,
                flags=re.IGNORECASE,
            )
            if not substantive:
                return "clarify"
        return "answer"

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
                if session.ticket_created:
                    return (
                        {"created": False, "error": "A handover ticket already exists for this session."},
                        {"name": name, "created": False, "duplicate_blocked": True},
                    )
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
