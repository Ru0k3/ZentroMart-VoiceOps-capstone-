# ZentroMart VoiceOps — practice report

**Author:** Individual project

**Repository:** https://github.com/Ru0k3/ZentroMart-VoiceOps-capstone-

**Dataset reference date:** 20 October 2026

**LLM used for the recorded full evaluation:** `nvidia/nemotron-3.5-lightning-30b-a3b`

## System

The implementation follows the four capstone stages: a prompt-only baseline,
section-aware RAG over the supplied knowledge base and product catalog,
multi-turn order and ticket tools, and Gnani Prisma STT → the support agent →
Gnani Timbre TTS. Retrieval uses BM25, retains section IDs and policy metadata,
and excludes superseded policies by default. CSV identifiers stay strings.
Order details are gated by the dataset verification tool; ticket success is
reported only after the tool confirms creation. Voice outputs save the reply,
WAV audio, retrieved evidence, and citations, but do not save the recognized
customer transcript.

## Full practice evaluation

The full saved run contains 52 baseline questions, 52 RAG questions, and 96
conversation turns. The model was `nvidia/nemotron-3.5-lightning-30b-a3b`.
Strict pass requires the expected decision, required and forbidden phrase
checks, expected tool actions, and all gold evidence IDs where applicable.
Phrase matching is literal after normalization, so these are practice-set
signals rather than semantic quality scores.

| Stage | Cases | Strict pass | Decision pass | Required facts pass | Forbidden facts pass | Gold evidence matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only baseline | 52 | 2 (3.8%) | 31/52 | 10/52 | 52/52 | 0/90 |
| RAG | 52 | 8 (15.4%) | 39/52 | 26/52 | 52/52 | 30/90 (33.3%) |
| Conversations | 96 | 2 (2.1%) | 54/96 | 72/96 | 96/96 | 2/112 (1.8%) |

RAG improved required-fact matches from 10 to 26 and matched 30 gold evidence
IDs where baseline matched none. The full conversation run passed 83/96
expected tool-action checks. Its traces showed parallel action proposals, so
the agent now executes at most one support action per customer turn and replaces
an order-specific answer with a verification request after a failed lookup. A
scripted local check confirmed those safeguards. Decision tags and source tags
are also recovered when omitted: decisions use a conservative text rule, and
citations require substantial overlap with retrieved passage content.

The final conversation run used the updated retrieval and prompt code and
completed all 96 turns, with explicit waits after NVIDIA HTTP 429 responses.
Its strict score was 2/96; evidence matching was 2/112 (1.8%). These low scores
are material limitations: grounded voice success on one targeted warranty case
does not imply reliable citation coverage across the full practice suite.
The main failure is evidence coverage in conversation mode. I would next test
retrieval depth and prompt behavior separately, then make citation selection
more reliable from the passages actually supplied to the model and rerun all
conversation cases. The low strict-pass counts also need case-by-case review;
the current literal phrase checks can miss correct paraphrases, but that does
not explain away the evidence-recall result.

**Targeted retrieval regression (PQ0002):** The practice ASR wording
“ele-zero zero one to” previously mapped poorly and unrelated product rows
crowded the relevant evidence out. Retrieval now normalizes spoken SKU digits,
boosts exact SKU and section-title matches, and excludes other product rows
when the query names an exact SKU. The targeted RAG case then retrieved both
`products.csv:ELE-0012` and `WAR-2` and matched its required facts. This does
not offset the full-set RAG score above. The result identifies a retrieval
robustness gap; it does not establish the dataset authors' intent.

## Voice run and limits

Gnani Timbre v2.5 with the Kaveri voice generated a valid 8.16-second WAV in a
TTS smoke run. A 62.55-second own-voice recording was converted to a 59.59-second,
16 kHz mono WAV; Gnani REST rejected it above its 30-second limit, so the runner
falls back to Gnani batch STT for that specific error. STT, RAG, and TTS then
completed once and produced a 28.96-second WAV.

The initial voice answer retrieved no `WAR-2` and had no citation. After the
Hindi/Hinglish retrieval changes and warranty-intent boost, a later own-voice
run retrieved `WAR-2` plus `products.csv:ELE-0012`, answered with the 12-month
Assured warranty and My Orders claim path, and generated a 25.76-second mono
48 kHz WAV. The citations are in the saved evidence metadata and are not read
aloud. This verifies one relevant voice case; it does not establish broad
multilingual accuracy. Only the builder's own voice and the supplied synthetic
dataset were used.

## Reproduction

Install dependencies with `pip install -r requirements.txt` and
`pip install -e .`; copy `.env.example` to `.env` and supply local keys. The
evaluation JSONL files and generated audio are local outputs and are not part
of the repository. See `DEMO_PLAN.md` for the short demonstration sequence.
