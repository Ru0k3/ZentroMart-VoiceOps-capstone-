# ZentroMart VoiceOps

An individual VoiceOps 2026 capstone project: a grounded ZentroMart customer
support assistant that uses company documents, verified mock order lookups, and
human handovers. The final stage adds Gnani speech-to-text and text-to-speech.

## Project status

See [PLAN.md](PLAN.md) for the stages and their acceptance criteria. The
dataset was validated locally, but the ZIP and extracted dataset are
deliberately excluded from Git as required by the capstone brief. Section-aware
KB parsing and BM25 retrieval live in
`src/zentromart/knowledge.py`. CSV loading, date rules, and guarded access to
the supplied mock tools, staged agent modes, and NVIDIA Build client live
alongside it under `src/zentromart/`. The full practice suites and one
end-to-end voice run have been exercised; see [REPORT.md](REPORT.md) for scores
and known gaps. Conversation safety changes and a Hindi retrieval fix were
made after those runs and still need provider-backed reruns.

## Dataset setup

Obtain `Capstone_Project_Dataset_FTTT_2026.zip` from the capstone organizers,
place it in this repository root, and extract it here. The expected directory
is `Capstone_Project_Dataset_FTTT_2026/`. Read its `README.md` and
`DATA_DICTIONARY.md` before working with the files. Run
`python validate_dataset.py` from that extracted directory to check the pack.

The dataset is synthetic. Do not use real customer data or another person's
voice recording. Use 20 October 2026 as its reference date; business-day counts run Monday
through Saturday except listed holidays, while return windows use calendar
days.

## Install and configure

Install Python dependencies and the project, then copy `.env.example` to `.env`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `NVIDIA_API_KEY`, `NVIDIA_MODEL`, and `GNANI_API_KEY` in `.env`. Keep that
file local; it is ignored by Git. Never put API keys in source code, commits,
reports, or chat. The same Gnani key is used for both STT and TTS.

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

The repository includes installation instructions, `.env.example`,
and one command that processes a folder of audio and saves text, speech, and
retrieved evidence. Deliverables also include a report of at most four pages,
sample outputs, and the [119-second individual demo](demo.mp4). See
[DEMO_PLAN.md](DEMO_PLAN.md) for its timings and replay instructions.

## Voice batch command

Place the builder's own test recordings in a folder. Supported input formats
are WAV, MP3, FLAC, OGG, and M4A; each clip must be at most 60 seconds. Run:

```bash
zentromart-voice ./my_audio --output ./outputs/voice_batch --mode conversation --language en-IN --tts-language en-IN --voice Kaveri
```

The command writes one subfolder per input clip with `reply.txt`, `reply.wav`,
and `evidence.json` containing retrieved chunks, cited IDs, decision, safe tool
events, and token use. It does not write the recognized customer transcript.
The default output directory is ignored by Git. Each file starts a fresh
conversation; use one customer utterance per file. Add `--session-scope folder`
when the ordered files are follow-up turns from one caller. Keep separate
callers in separate runs. `--mode` also accepts `baseline` and `rag`. Gnani is
the only speech provider.

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
