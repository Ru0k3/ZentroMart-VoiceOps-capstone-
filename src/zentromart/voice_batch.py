"""Batch voice wrapper: Gnani STT -> the existing support agent -> Gnani TTS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

from .agent import AgentSession
from .factory import create_agent
from .llm import load_local_env


AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
RECOVERY_REPLY = "Sorry, I didn't catch that. Could you say it again?"


def prepare_for_speech(text: str) -> str:
    """Format order numbers for spoken playback without changing other text."""
    return re.sub(r"(?<!\d)(\d{4})(\d{4})(?!\d)", r"\1 \2", text)


def normalize_wav_header(audio_bytes: bytes) -> bytes:
    """Repair unknown-length RIFF sizes some TTS streams leave as 0xFFFFFFFF."""
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError("Gnani TTS returned data without a RIFF/WAVE header")
    if len(audio_bytes) - 8 >= 0xFFFFFFFF:
        raise ValueError("Gnani TTS WAV is too large for a standard RIFF container")

    repaired = bytearray(audio_bytes)
    repaired[4:8] = (len(repaired) - 8).to_bytes(4, "little")
    offset = 12
    found_data = False
    while offset + 8 <= len(repaired):
        chunk_id = repaired[offset:offset + 4]
        chunk_size = int.from_bytes(repaired[offset + 4:offset + 8], "little")
        content_start = offset + 8
        if chunk_size == 0xFFFFFFFF:
            if chunk_id != b"data":
                raise ValueError("Gnani TTS WAV has an unknown non-audio chunk size")
            chunk_size = len(repaired) - content_start
            repaired[offset + 4:offset + 8] = chunk_size.to_bytes(4, "little")
        if content_start + chunk_size > len(repaired):
            raise ValueError("Gnani TTS WAV contains a truncated chunk")
        if chunk_id == b"data":
            found_data = True
        offset = content_start + chunk_size + (chunk_size % 2)

    if not found_data:
        raise ValueError("Gnani TTS WAV does not contain an audio data chunk")
    return bytes(repaired)


def transcribe_audio(stt: Any, source: Path, language: str) -> dict[str, Any]:
    """Use Gnani batch STT when REST rejects a clip over its duration limit."""
    try:
        return stt.transcribe(str(source), language_code=language)
    except Exception as error:
        try:
            from gnani.stt import APIError, GnaniSTTBatchClient
        except ImportError:
            raise error
        if not isinstance(error, APIError) or error.status_code != 400:
            raise
        try:
            response = json.loads(error.body)
            detail = response.get("error", {})
            error_type = detail.get("type") if isinstance(detail, dict) else None
        except (TypeError, ValueError):
            error_type = None
        if error_type != "MAX_AUDIO_DURATION_EXCEEDED":
            raise

        result = GnaniSTTBatchClient().transcribe(
            [str(source)], language_code=language, progress=False
        )
        if len(result) != 1 or not result[0].ok:
            raise RuntimeError("Gnani batch STT could not transcribe the input audio")
        return {"transcript": result[0].text}


def find_audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Audio input folder does not exist: {folder}")
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
    )


def _clients() -> tuple[Any, Any]:
    load_local_env()
    if not os.environ.get("GNANI_API_KEY", "").strip():
        raise RuntimeError("Set GNANI_API_KEY in the local environment or .env")
    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
    except ImportError as error:
        raise RuntimeError(
            "Gnani SDK is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from error
    return GnaniSTTClient(), GnaniTTSClient()


def _relative_target(source: Path, input_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(input_root)
    return output_root / relative.parent / f"{relative.stem}-{relative.suffix[1:].casefold()}"


def process_audio_file(
    source: Path,
    input_root: Path,
    output_root: Path,
    *,
    agent: Any,
    stt: Any,
    tts: Any,
    session: AgentSession,
    mode: str,
    language: str,
    voice: str,
    tts_language: str,
    tts_speed: float,
) -> dict[str, Any]:
    target = _relative_target(source, input_root, output_root)
    target.mkdir(parents=True, exist_ok=True)

    transcription = transcribe_audio(stt, source, language)
    transcript = str(transcription.get("transcript", "")).strip()
    if transcript:
        turn = agent.respond(transcript, session)
        reply = turn.answer
    else:
        turn = None
        reply = RECOVERY_REPLY

    speech_text = prepare_for_speech(reply)
    audio_bytes = tts.synthesize(
        speech_text,
        voice=voice,
        model="timbre-v2.5",
        language=tts_language,
        speed=tts_speed,
    )
    audio_bytes = normalize_wav_header(audio_bytes)
    (target / "reply.txt").write_text(speech_text + "\n", encoding="utf-8")
    (target / "reply.wav").write_bytes(audio_bytes)

    evidence = {
        "source_audio": source.relative_to(input_root).as_posix(),
        "mode": mode,
        "stt_language": language,
        "tts_model": "timbre-v2.5",
        "tts_voice": voice,
        "reply": speech_text,
        "decision": turn.decision if turn else "clarify",
        "retrieved_evidence": turn.retrieved_evidence if turn else [],
        "cited_evidence_ids": turn.cited_evidence_ids if turn else [],
        "tool_calls": turn.tool_calls if turn else [],
        "usage": turn.usage if turn else {},
    }
    (target / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"source": source.relative_to(input_root).as_posix(), "status": "ok", "output": str(target)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run each audio file through Gnani STT, ZentroMart, and Gnani TTS."
    )
    parser.add_argument("audio_folder", type=Path, help="Folder of WAV, MP3, FLAC, OGG, or M4A files")
    parser.add_argument("--output", type=Path, default=Path("outputs/voice_batch"))
    parser.add_argument("--mode", choices=("baseline", "rag", "conversation"), default="conversation")
    parser.add_argument(
        "--session-scope",
        choices=("file", "folder"),
        default="file",
        help="Start a fresh session per file, or share one session across this folder.",
    )
    parser.add_argument("--language", default="en-IN", help="Gnani STT language code, for example en-IN or hi-IN")
    parser.add_argument("--tts-language", default="en-IN", help="Timbre v2.5 language code")
    parser.add_argument("--voice", default="Kaveri", help="Gnani Timbre voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed, 0.85–1.15")
    args = parser.parse_args(argv)
    if not 0.85 <= args.speed <= 1.15:
        parser.error("--speed must be between 0.85 and 1.15")

    try:
        input_root = args.audio_folder.expanduser().resolve(strict=True)
        output_root = args.output.expanduser().resolve()
        if (
            output_root == input_root
            or input_root in output_root.parents
            or output_root in input_root.parents
        ):
            parser.error("Input and output folders must be separate, non-overlapping directories")
        files = find_audio_files(input_root)
        if not files:
            parser.error(f"No supported audio files found under {input_root}")
        stt, tts = _clients()
        agent = create_agent(args.mode)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, OSError) as error:
        parser.error(str(error))

    failures = 0
    folder_session = agent.new_session()
    for source in files:
        session = folder_session if args.session_scope == "folder" else agent.new_session()
        try:
            outcome = process_audio_file(
                source,
                input_root,
                output_root,
                agent=agent,
                stt=stt,
                tts=tts,
                session=session,
                mode=args.mode,
                language=args.language,
                voice=args.voice,
                tts_language=args.tts_language,
                tts_speed=args.speed,
            )
        except Exception as error:  # Continue the batch; never print request data or credentials.
            failures += 1
            outcome = {
                "source": source.relative_to(input_root).as_posix(),
                "status": "error",
                "error": type(error).__name__,
            }
        print(json.dumps(outcome, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
