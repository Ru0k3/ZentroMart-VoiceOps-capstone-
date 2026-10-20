"""Guarded access to the identity-checked mock tools shipped with the dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import re
from types import ModuleType
from typing import Any

from .dataset import Dataset


_ASCII_DIGITS = re.compile(r"[^0-9]")
_ORDER_ID_PATTERN = re.compile(r"(?<!\d)\d{4}(?:[\s-]*\d{4})(?!\d)")


def _load_dataset_tools(dataset_root: Path) -> ModuleType:
    path = dataset_root / "tools" / "tools.py"
    spec = importlib.util.spec_from_file_location("zentromart_capstone_tools", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load provided capstone tools at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ConversationToolGuard:
    """Apply conversation-level verification limits around the supplied tools.

    The helper tool enforces a match before returning order data. This wrapper
    additionally ensures proof came from a caller message in this turn, allows
    at most one retry after an initial failed verification, and redacts proof
    from handover summaries.
    """

    dataset: Dataset
    _attempts: dict[str, int] = field(default_factory=dict, init=False)
    _verified: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _proof_values: set[str] = field(default_factory=set, init=False)
    _seen_order_ids: set[str] = field(default_factory=set, init=False)
    _tools: ModuleType = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tools = _load_dataset_tools(self.dataset.root)

    def record_caller_message(self, caller_message: str) -> None:
        """Record order IDs mentioned by the caller before handling tool calls."""
        for match in _ORDER_ID_PATTERN.finditer(caller_message):
            self._seen_order_ids.add(_ASCII_DIGITS.sub("", match.group(0)))

    def lookup_order(
        self,
        order_id: str,
        caller_message: str,
        verification_value: str | None = None,
    ) -> dict[str, Any]:
        order_key = _ASCII_DIGITS.sub("", str(order_id))
        self.record_caller_message(caller_message)
        if order_key not in self._seen_order_ids:
            return {
                "verification_required": True,
                "message": "Ask the caller for their order ID before looking it up.",
            }
        if order_key in self._verified:
            return self._verified[order_key].copy()

        if not verification_value:
            return self._tools.lookup_order(order_key)

        proof = _ASCII_DIGITS.sub("", str(verification_value))
        caller_digits = _ASCII_DIGITS.sub("", caller_message)
        if not proof or proof not in caller_digits:
            return {
                "verification_required": True,
                "message": "Ask the caller for the registered phone last 4 digits or delivery pincode.",
            }

        attempts = self._attempts.get(order_key, 0)
        if attempts >= 2:
            return {
                "verified": False,
                "verification_required": False,
                "message": "Verification attempts exhausted; do not share order details.",
            }

        self._proof_values.add(proof)
        result = self._tools.lookup_order(order_key, verification_value=proof)
        if result.get("verified"):
            self._verified[order_key] = result.copy()
        else:
            self._attempts[order_key] = attempts + 1
        return result

    def create_ticket(self, **arguments: Any) -> dict[str, Any]:
        safe_arguments = {
            key: value for key, value in arguments.items()
            if key in {"summary", "order_id", "reason", "intent"}
        }
        summary = str(safe_arguments.get("summary", ""))
        for proof in self._proof_values:
            summary = re.sub(
                rf"(?<!\d){re.escape(proof)}(?!\d)",
                "[verification redacted]",
                summary,
            )
        safe_arguments["summary"] = summary
        return self._tools.create_ticket(**safe_arguments)

    def has_verified(self, order_id: str) -> bool:
        order_key = _ASCII_DIGITS.sub("", str(order_id))
        return order_key in self._verified
