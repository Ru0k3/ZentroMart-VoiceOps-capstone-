"""Construct configured agent stages from the local dataset and environment."""

from __future__ import annotations

from pathlib import Path

from .agent import AgentMode, ZentroMartAgent
from .dataset import Dataset
from .knowledge import (
    KnowledgeRetriever,
    load_knowledge_base,
    product_catalog_chunks,
)
from .llm import NVIDIAChatClient


def create_agent(
    mode: AgentMode = "conversation",
    *,
    dataset_root: Path | str | None = None,
    top_k: int = 5,
) -> ZentroMartAgent:
    """Load the local evidence and build an agent without making an API call."""
    dataset = Dataset(dataset_root)
    kb_root = dataset.root / "kb"
    chunks = load_knowledge_base(kb_root)
    chunks.extend(product_catalog_chunks(dataset.table("products")))
    retriever = KnowledgeRetriever(chunks)
    client = NVIDIAChatClient.from_environment()
    return ZentroMartAgent(
        client,
        dataset,
        retriever,
        mode=mode,
        top_k=top_k,
    )
