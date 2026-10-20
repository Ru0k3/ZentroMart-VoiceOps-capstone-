# ZentroMart VoiceOps

An individual VoiceOps 2026 capstone project: a grounded ZentroMart customer
support assistant that uses company documents, verified mock order lookups, and
human handovers. The final stage adds Gnani speech-to-text and text-to-speech.

## Project status

The build is being delivered in parts. See [PLAN.md](PLAN.md) for the stages
and their acceptance criteria. The dataset was validated locally, but the ZIP
and extracted dataset are deliberately excluded from Git as required by the
capstone submission brief. Section-aware KB parsing and BM25 retrieval live in
`src/zentromart/knowledge.py`. CSV loading, date rules, and guarded access to
the supplied mock tools, staged agent modes, and NVIDIA Build client live
alongside it under `src/zentromart/`. Model requests are not sent until an
NVIDIA model and local credentials are configured.

## Dataset setup

Obtain `Capstone_Project_Dataset_FTTT_2026.zip` from the capstone organizers,
place it in this repository root, and extract it here. The expected directory
is `Capstone_Project_Dataset_FTTT_2026/`. Read its `README.md` and
`DATA_DICTIONARY.md` before working with the files. Run
`python validate_dataset.py` from that extracted directory to check the pack.

The dataset is synthetic. Do not substitute real customer data or recordings.
Use 20 October 2026 as its reference date; business-day counts run Monday
through Saturday except listed holidays, while return windows use calendar
days.

## Secrets and providers

Copy `.env.example` to `.env` for local development and keep the latter out of
Git. The NVIDIA Build model will be selected during implementation. Gnani STT
and TTS credentials are only needed for the final voice stage. Never put API
keys in source code, commits, reports, or chat.

## Capstone requirements

- Answer from retrieved policy evidence; do not hardcode practice answers.
- Preserve section IDs and report which chunks support each response.
- Keep CSV IDs and pincodes as strings.
- Verify a caller before revealing specific order details.
- Confirm a human handover only after ticket creation succeeds.
- Use Gnani for both speech-to-text and text-to-speech; no other speech
  provider is allowed.
- The submission must not contain the dataset copy, secrets, or a virtual
  environment.

## Submission target

The final repository will include installation instructions, `.env.example`,
and one command that processes a folder of audio and saves text, speech, and
retrieved evidence. Deliverables also include a report of at most four pages,
sample outputs, and an individual demo of at most two minutes.

## Practice evaluation

Run a small RAG sample after configuring the NVIDIA Build key and model:

```bash
PYTHONPATH=src python -m zentromart.evaluation --mode rag --limit 1
```

The evaluator reports decisions, literal required/forbidden phrase checks,
evidence coverage, tool-call expectations, and token usage. Fact matching is
an approximate phrase check; review multilingual and paraphrased cases by hand.
The JSONL output is written under the ignored `outputs/` directory and omits
customer messages and verification values. Use `--mode all --all-cases` to run
the complete baseline, RAG, and multi-turn practice suites; this can make many
NVIDIA requests and consume credits.
