# Two-minute demo

`demo.mp4` is a 119.35-second demonstration using the builder's own Hindi
recording, the verified Gnani TTS reply, retrieved source IDs, a warranty
follow-up evidence check, and an actual unsupported practice case. It uses only
the supplied synthetic dataset. The private input recording and recognized
transcript are not included in the repository.

## Video timing

- **0:00–0:04** — Project title and stages.
- **0:04–1:04** — Own-voice Hindi input through Gnani Prisma STT.
- **1:04–1:30** — Gnani Timbre answer about SKU `ELE-0012` and its Assured
  warranty.
- **1:30–1:40** — Follow-up evidence check: `WAR-3` excludes liquid damage.
- **1:40–1:50** — Show `WAR-2` and `products.csv:ELE-0012`, which support the
  spoken answer.
- **1:50–1:59** — Unsupported midnight-delivery practice question; the retrieved
  material does not establish available time slots.

## Live demo commands

To replay the audio flow with another own-voice recording, put the clip alone in
a folder and run:

```bash
zentromart-voice path/to/own-voice-clip --output outputs/demo --mode rag --language hi-IN --tts-language hi-IN --voice Kaveri
```

Show `reply.txt`, play `reply.wav`, then compare `cited_evidence_ids` in
`evidence.json` to the retrieved passages. For the follow-up retrieval check,
ask whether an Assured warranty covers liquid damage; confirm that `WAR-3`
appears and supports the answer. For order-specific conversation turns, use
only synthetic dataset records and show no details before successful identity
verification. Cite the evaluation limitations in `REPORT.md`; do not imply that
one voice case represents overall model performance.
