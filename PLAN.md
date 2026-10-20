# Build plan

This individual capstone follows the four steps in `capstone_project.pdf` and
the required dataset workflow. Each completed part will be committed and pushed
before work continues to the next part.

## Part 1 — Project foundation and dataset contract

- Read the dataset README and data dictionary, then run its validator.
- Define project setup, local secret handling, and exclusions for the dataset
  and generated artifacts.
- Record the implementation stages and acceptance criteria.
- Status: complete; dataset validator passes and the project foundation is in place.

## Part 2 — Knowledge-base ingestion and retrieval

- Parse the 21 Markdown files and preserve document metadata, section IDs,
  policy status, and effective dates.
- Apply the documented source precedence and exclude superseded material from
  normal policy answers unless its successor says otherwise.
- Retrieve and retain source chunks for every answer.
- Acceptance: retrieval returns ranked section chunks with their evidence IDs
  and metadata; superseded chunks are excluded by default. Answer abstention
  will be implemented with the LLM agent in Part 4.
- Status: complete; section parser and dependency-free BM25 retrieval are in
  `src/zentromart/knowledge.py`.

## Part 3 — Structured data, tools, and conversation handling

- Load CSVs with identifiers preserved as strings and documented joins.
- Use the supplied identity-gated `lookup_order` and `create_ticket` tools.
- Add multi-turn context, clarifying questions, one-retry verification, policy
  refusals, and confirmed human handovers.
- Acceptance: no order details before verification; no claimed ticket unless
  the tool reports `created=True`.

## Part 4 — Text baseline and evaluation

- Implement the prompt-only baseline and the RAG agent as separate stages.
- Evaluate all practice questions and conversations, including ASR-like text,
  expected decisions, tool calls, and evidence IDs.
- Report the Step 1 vs. Step 2 comparison and known failures.
- NVIDIA model choice is intentionally deferred until the available Build
  models and access are confirmed.

## Part 5 — Gnani voice and reproducible submission

- Add Gnani-only STT and TTS after an API key is available; do not add a
  different speech provider.
- Provide one batch command that accepts an audio folder and saves text,
  speech, and evidence chunks.
- Prepare `.env.example`, setup instructions, a report (maximum four pages),
  sample outputs, and an individual two-minute demo plan/video.
- Acceptance: no keys, dataset copy, virtual environment, or real customer data
  in the submission.

## Dataset rules carried into implementation

- Use 20 October 2026 as the fixed reference date.
- Business days are Monday–Saturday, excluding listed holidays; return windows
  use calendar days.
- Preserve IDs and pincodes as strings; blank fields mean missing or
  inapplicable.
- Policies outrank agent guides, help, FAQs, and marketing; human support logs
  are not policy authority.
- Use only the synthetic dataset and the builder's own voice for voice testing.
