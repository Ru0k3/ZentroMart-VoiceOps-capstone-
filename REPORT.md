# ZentroMart VoiceOps — capstone report draft

**Author:** Individual project  
**Repository:** https://github.com/Ru0k3/ZentroMart-VoiceOps-capstone-  
**Dataset date:** 20 October 2026

## What I built

The project follows the four capstone stages. Step 1 is a prompt-only baseline.
Step 2 retrieves section-aware passages from the supplied knowledge base and
product catalog. Step 3 adds multi-turn context and identity-gated order and
ticket tools. Step 4 sends audio through Gnani Prisma STT, the same support
agent, then Gnani Timbre TTS. NVIDIA Build is the configurable LLM provider.

The retrieval layer preserves section IDs and policy metadata, ranks evidence
with BM25, and excludes superseded material by default. CSV values remain
strings. Order details require the supplied verification tool, and a handover
is reported only after the ticket tool confirms creation.

## Evaluation

Run the supplied practice questions in both baseline and RAG modes, then run
the multi-turn conversations. The harness records expected decisions, literal
required and forbidden phrase checks, cited evidence, tool-call expectations,
and model token counts. Phrase matching is approximate; multilingual cases
and paraphrases require human review.

| Practice suite | Cases | Decision pass | Evidence recall | Strict pass |
| --- | ---: | ---: | ---: | ---: |
| Prompt-only baseline | Pending run | Pending | n/a | Pending |
| RAG | Pending run | Pending | Pending | Pending |
| Multi-turn conversations | Pending run | Pending | Pending | Pending |

**What failed and what I would improve:** Complete this section after reviewing
the actual run output. Do not enter scores until the harness has run with the
selected NVIDIA model and available credit budget.

## Voice and reproducibility

The batch command accepts WAV, MP3, FLAC, OGG, and M4A clips up to 60 seconds.
For each clip it saves the spoken reply text, Gnani-generated WAV, and retrieved
evidence metadata. It does not persist recognized customer transcripts. Only
Gnani is used for speech recognition and synthesis.

Install with `pip install -r requirements.txt` and `pip install -e .`; configure
keys in a local `.env` copied from `.env.example`. The dataset and generated
outputs are excluded from the repository.

## Limitations and next steps

Complete this section after the live run: record model/version, voice choice,
observed errors, the Step 1 vs. Step 2 comparison, and any changes made after
the test. Keep the final exported PDF within four pages.
