"""Speech-to-text providers that label speakers.

Each provider takes a 16 kHz mono FLAC (already converted by the caller) and
returns speaker-labelled segments. Each one is called with its privacy
settings on, see the comment at the top of each class.
"""

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import HTTPException

TIMEOUT = httpx.Timeout(900, connect=30)


@dataclass
class Segment:
    speaker: int
    start: float | None  # None when the provider gives no timestamps
    end: float | None
    text: str


class SpeakerNumbers:
    """Maps provider labels ("spk_1", "A", 3) to 0, 1, 2… in order of first appearance."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def __call__(self, label: object) -> int:
        return self._ids.setdefault(str(label), len(self._ids))


def merge_turns(turns: list[Segment]) -> list[Segment]:
    """Join consecutive turns by the same speaker into one paragraph."""
    merged: list[Segment] = []
    for t in turns:
        t.text = t.text.strip()
        if not t.text:
            continue
        if merged and merged[-1].speaker == t.speaker:
            merged[-1].end = t.end
            merged[-1].text += " " + t.text
        else:
            merged.append(t)
    return merged


async def ffmpeg(src: Path, dest: Path, *codec: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", *codec, str(dest),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    _, err = await proc.communicate()
    if proc.returncode != 0:
        detail = err.decode(errors="replace").strip()[-300:]
        raise HTTPException(422, "Could not read this file as audio: " + detail)


def provider_error(name: str, resp: httpx.Response) -> HTTPException:
    return HTTPException(502, f"{name} error {resp.status_code}: {resp.text[:400]}")


class Deepgram:
    """Nova-3 with the v2 speaker model. mip_opt_out=true opts out of Deepgram's
    Model Improvement Program, so the audio is not kept for training."""

    id, label = "deepgram", "Deepgram Nova-3"
    key_env = "DEEPGRAM_API_KEY"
    max_seconds = None
    url = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, flac: Path, workdir: Path, language: str) -> list[Segment]:
        params = {
            "model": "nova-3",
            "language": language,
            "diarize_model": "latest",  # v2+ speaker model; plain diarize=true pins v1
            "utterances": "true",
            "smart_format": "true",
            "punctuate": "true",
            "mip_opt_out": "true",
        }
        headers = {
            "Authorization": f"Token {os.environ[self.key_env]}",
            "Content-Type": "audio/flac",
        }
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                self.url, params=params, content=flac.read_bytes(), headers=headers
            )
        if resp.status_code != 200:
            raise provider_error("Deepgram", resp)
        return self.parse(resp.json())

    @staticmethod
    def parse(result: dict) -> list[Segment]:
        n = SpeakerNumbers()
        return merge_turns([
            Segment(n(u.get("speaker", 0)), u["start"], u["end"], u.get("transcript", ""))
            for u in result.get("results", {}).get("utterances", [])
        ])  # fmt: skip


class Gemini:
    """Gemini 3.5 Transcribe. The audio is sent inline (no Files API upload, which
    Google would keep for 48 hours) and store=false stops the Interactions API
    keeping the request (55 days by default). Google only excludes data from
    training on the paid tier, so the key's project must have billing on."""

    id, label = "gemini", "Gemini 3.5 Transcribe"
    key_env = "GEMINI_API_KEY"
    # With speaker labels Gemini processes 30 minutes and silently truncates the rest.
    max_seconds = 30 * 60 + 1
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    languages = {"en": "en-US", "es": "es-ES", "de": "de-DE", "fr": "fr-FR", "it": "it-IT",
                 "pt": "pt-PT", "nl": "nl-NL", "ru": "ru-RU", "multi": None}  # fmt: skip

    async def transcribe(self, flac: Path, workdir: Path, language: str) -> list[Segment]:
        code = self.languages.get(language)
        body = {
            "model": "models/gemini-3.5-transcribe",
            "input": [
                {
                    "type": "audio",
                    "mime_type": "audio/flac",
                    "data": base64.b64encode(flac.read_bytes()).decode(),
                }
            ],  # fmt: skip
            "generation_config": {
                "transcription_config": {
                    "mode": {"type": "verbatim", "diarization_mode": "speaker"},
                    "language_codes": [code] if code else [],
                }
            },
            "store": False,
        }
        headers = {"x-goog-api-key": os.environ[self.key_env]}
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(self.url, json=body, headers=headers)
        if resp.status_code != 200:
            raise provider_error("Gemini", resp)
        return self.parse(resp.json())

    @staticmethod
    def parse(result: dict) -> list[Segment]:
        n = SpeakerNumbers()
        turns: list[Segment] = []
        for step in result.get("steps", []):
            for content in step.get("content", []):
                for a in content.get("annotations", []):
                    if a.get("type") != "word_info":
                        continue
                    start, end = seconds(a.get("start_offset")), seconds(a.get("end_offset"))
                    turns.append(Segment(n(a.get("speaker", "")), start, end, a.get("text", "")))
        return merge_turns(turns)


def seconds(offset: str | None) -> float | None:
    """'12.340s' -> 12.34"""
    m = re.fullmatch(r"([\d.]+)s", offset or "")
    return float(m.group(1)) if m else None


class OpenAI:
    """gpt-4o-transcribe-diarize, OpenAI's only model that labels speakers; it is
    deprecated and shuts down on 2027-02-26. API data is not used for training."""

    id, label = "openai", "OpenAI GPT-4o Transcribe Diarize"
    key_env = "OPENAI_API_KEY"
    max_seconds = None
    max_bytes = 25 * 1024 * 1024
    url = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(self, flac: Path, workdir: Path, language: str) -> list[Segment]:
        # 25 MB upload cap: mono Opus at 32 kbps is ~14 MB/hour and fine for speech.
        ogg = workdir / "audio.ogg"
        await ffmpeg(flac, ogg, "-c:a", "libopus", "-b:a", "32k", "-application", "voip")
        if ogg.stat().st_size > self.max_bytes:
            raise HTTPException(413, "Too long for OpenAI (25 MB limit, about 100 minutes)")
        data = {
            "model": "gpt-4o-transcribe-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }
        if language != "multi":
            data["language"] = language
        headers = {"Authorization": f"Bearer {os.environ[self.key_env]}"}
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                self.url, data=data, headers=headers,
                files={"file": ("audio.ogg", ogg.read_bytes(), "audio/ogg")},
            )  # fmt: skip
        if resp.status_code != 200:
            raise provider_error("OpenAI", resp)
        return self.parse(resp.json())

    @staticmethod
    def parse(result: dict) -> list[Segment]:
        n = SpeakerNumbers()
        return merge_turns([
            Segment(n(s.get("speaker", "")), s.get("start"), s.get("end"), s.get("text", ""))
            for s in result.get("segments", [])
        ])  # fmt: skip


ALL = [Deepgram(), Gemini(), OpenAI()]


def available() -> dict:
    """Providers whose API key is set, in display order."""
    return {p.id: p for p in ALL if os.environ.get(p.key_env)}
