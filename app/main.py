"""Transcribe: upload a recording, get a speaker-labelled transcript.

Audio goes to Deepgram's pre-recorded (batch) API with diarization on and the
Model Improvement Program opted out, so Deepgram does not keep it for
training. Nothing is written to disk beyond a per-request temp dir that is
deleted when the request ends, and nothing is logged except paths and status.
"""

import asyncio
import io
import os
import secrets
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from docx import Document
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel

load_dotenv()

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
LANGUAGES = {"en", "es", "de", "fr", "it", "pt", "nl", "ru", "multi"}


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set; see .env.example")
    return value


DEEPGRAM_API_KEY = _env("DEEPGRAM_API_KEY")
ACCESS_TOKEN = _env("ACCESS_TOKEN")
if len(ACCESS_TOKEN) < 20:
    raise RuntimeError("ACCESS_TOKEN must be at least 20 characters; see .env.example")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "en")
COOKIE = "access"
COOKIE_MAX_AGE = 365 * 24 * 3600

NO_ACCESS_PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Transcribe</title>
<body style="font:16px system-ui,sans-serif;max-width:560px;margin:15vh auto;padding:0 16px">
<h1>Transcribe</h1><p>This page is private. Open it with the link you were given.</p></body>"""


def has_access(request: Request) -> bool:
    cookie = request.cookies.get(COOKIE, "")
    return secrets.compare_digest(cookie.encode(), ACCESS_TOKEN.encode())


async def require_access(request: Request) -> None:
    if not has_access(request):
        raise HTTPException(401, "This page is private. Open it with the link you were given.")


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
api = APIRouter(prefix="/api", dependencies=[Depends(require_access)])


@app.middleware("http")
async def no_referrer(request: Request, call_next):
    response = await call_next(request)
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/s/{token}")
async def open_link(token: str, request: Request) -> Response:
    """The shareable link: swap the token for a cookie, then drop it from the URL."""
    if not secrets.compare_digest(token.encode(), ACCESS_TOKEN.encode()):
        await asyncio.sleep(1)  # slow down guessing
        return HTMLResponse(NO_ACCESS_PAGE, status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        COOKIE,
        ACCESS_TOKEN,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )
    return response


@dataclass
class Segment:
    speaker: int
    start: float
    end: float
    text: str


@app.get("/")
async def index(request: Request) -> Response:
    if not has_access(request):
        return HTMLResponse(NO_ACCESS_PAGE, status_code=401)
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@api.get("/config")
async def config() -> dict:
    return {"default_language": DEFAULT_LANGUAGE, "max_upload_mb": MAX_UPLOAD_BYTES >> 20}


async def save_upload(upload: UploadFile, dest: Path) -> None:
    size = 0
    with dest.open("wb") as out:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES >> 20} MB")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "The file is empty")


async def to_flac(src: Path, dest: Path) -> None:
    """Any audio or video in, 16 kHz mono FLAC out: lossless for speech, ~1 MB/min."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(dest),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(
            422, "Could not read this file as audio: " + err.decode(errors="replace").strip()[-300:]
        )


async def call_deepgram(audio: Path, language: str) -> dict:
    params = {
        "model": "nova-3",
        "language": language,
        "diarize_model": "latest",  # v2+ speaker model; plain diarize=true pins v1
        "utterances": "true",
        "smart_format": "true",
        "punctuate": "true",
        "mip_opt_out": "true",  # do not let Deepgram keep the audio for training
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as client:
        with audio.open("rb") as f:
            resp = await client.post(
                DEEPGRAM_URL,
                params=params,
                content=f.read(),
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/flac",
                },
            )
    if resp.status_code != 200:
        raise HTTPException(502, f"Deepgram error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def to_segments(result: dict) -> list[Segment]:
    """Deepgram utterances, with consecutive turns by the same speaker merged."""
    segments: list[Segment] = []
    for u in result.get("results", {}).get("utterances", []):
        text = u.get("transcript", "").strip()
        if not text:
            continue
        speaker = int(u.get("speaker", 0))
        if segments and segments[-1].speaker == speaker:
            segments[-1].end = u["end"]
            segments[-1].text += " " + text
        else:
            segments.append(Segment(speaker, u["start"], u["end"], text))
    return segments


@api.post("/transcribe")
async def transcribe(file: UploadFile, language: str = Form(DEFAULT_LANGUAGE)) -> dict:
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language {language!r}")
    workdir = Path(tempfile.mkdtemp(prefix="transcribe-"))
    try:
        src, flac = workdir / "upload", workdir / "audio.flac"
        await save_upload(file, src)
        await to_flac(src, flac)
        src.unlink()
        result = await call_deepgram(flac, language)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    segments = to_segments(result)
    return {
        "duration": result.get("metadata", {}).get("duration"),
        "speakers": sorted({s.speaker for s in segments}),
        "segments": [asdict(s) for s in segments],
    }


class DocxRequest(BaseModel):
    title: str
    names: dict[str, str]
    segments: list[dict]


def timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


@api.post("/docx")
async def docx(req: DocxRequest) -> StreamingResponse:
    doc = Document()
    doc.add_heading(req.title or "Transcript", level=1)
    for seg in req.segments:
        speaker = str(seg["speaker"])
        p = doc.add_paragraph()
        p.add_run(f"[{timestamp(seg['start'])}] ").italic = True
        p.add_run(f"{req.names.get(speaker) or 'Speaker ' + speaker}: ").bold = True
        p.add_run(seg["text"])
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="transcript.docx"'},
    )


app.include_router(api)
