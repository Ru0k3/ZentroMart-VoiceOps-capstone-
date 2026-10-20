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

## Part 3 — Structured data, date rules, and guarded tools

- Load CSVs with identifiers preserved as strings and documented joins.
- Use the supplied identity-gated `lookup_order` and `create_ticket` tools.
- Apply the fixed reference date and business-day rules.
- Guard tool use so the order ID is caller-provided, verification proof was
  stated by the caller, one retry is allowed, and proof is redacted from ticket
  summaries.
- Acceptance: no order details before successful verification; no claimed
  ticket unless the tool reports `created=True`.
- Status: complete; CSV, date, and tool-boundary modules are in `src/zentromart/`.

## Part 4 — NVIDIA LLM agent

- Implement the prompt-only baseline and the RAG agent as separate stages,
  using an NVIDIA Build model selected with the user.
- Add multi-turn context, clarifying questions, refusals, and guarded function
  calls.
- Keep model, API base, and key configurable; record token usage and avoid
  hidden retries.
- Acceptance: baseline, RAG, and conversation modes share one client; tool
  calls are bounded and verification values are not retained in session text.
- Status: code complete; NVIDIA API calls wait for model, key, and the credit
  budget clarification.

## Part 5 — Evaluation

- Evaluate all practice questions and conversations, including ASR-like text,
  expected decisions, tool calls, and evidence IDs.
- Report the Step 1 vs. Step 2 comparison and known failures.
- Running the paid/credited model evaluation waits until the model and available
  credit budget are clear.
- Status: harness implemented in `src/zentromart/evaluation.py`; model runs are
  pending the user's NVIDIA model and credit clarification.

## Part 6 — Gnani voice and reproducible submission

- Add Gnani-only STT and TTS after an API key is available; do not add a
  different speech provider.
- Provide one batch command that accepts an audio folder and saves text,
  speech, and evidence chunks.
- Prepare `.env.example`, setup instructions, a report (maximum four pages),
  sample outputs, and an individual two-minute demo plan/video.
- Acceptance: no keys, dataset copy, virtual environment, or real customer data
  in the submission.
- Status: Gnani REST STT/TTS batch pipeline, install metadata, command-line
  entry point, report draft, demo outline, and sample-output instructions are
  implemented. Live audio/model validation and actual sample outputs await
  the scheduled test; credentials remain local.

## Dataset rules carried into implementation

- Use 20 October 2026 as the fixed reference date.
- Business days are Monday–Saturday, excluding listed holidays; return windows
  use calendar days.
- Preserve IDs and pincodes as strings; blank fields mean missing or
  inapplicable.
- Policies outrank agent guides, help, FAQs, and marketing; human support logs
  are not policy authority.
- Use only the synthetic dataset and the builder's own voice for voice testing.
