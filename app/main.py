"""Transcribe: upload a recording, get a speaker-labelled transcript.

The upload is converted to 16 kHz mono FLAC and sent to the chosen provider
(see providers.py for each one's privacy settings). Nothing is written to disk
beyond a per-request temp dir that is deleted when the request ends, and
nothing is logged except paths and status.
"""

import asyncio
import io
import json
import os
import secrets
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

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

from app import providers

load_dotenv()

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
LANGUAGES = {"en", "es", "de", "fr", "it", "pt", "nl", "ru", "multi"}


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set; see .env.example")
    return value


PROVIDERS = providers.available()
if not PROVIDERS:
    keys = ", ".join(p.key_env for p in providers.ALL)
    raise RuntimeError(f"Set at least one provider key ({keys}); see .env.example")
ACCESS_TOKEN = _env("ACCESS_TOKEN")
if len(ACCESS_TOKEN) < 20:
    raise RuntimeError("ACCESS_TOKEN must be at least 20 characters; see .env.example")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "es")
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


@app.get("/")
async def index(request: Request) -> Response:
    if not has_access(request):
        return HTMLResponse(NO_ACCESS_PAGE, status_code=401)
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@api.get("/config")
async def config() -> dict:
    return {
        "default_language": DEFAULT_LANGUAGE,
        "max_upload_mb": MAX_UPLOAD_BYTES >> 20,
        "providers": [
            {"id": p.id, "label": p.label, "max_minutes": p.max_seconds and p.max_seconds // 60}
            for p in PROVIDERS.values()
        ],
    }


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


async def duration_of(path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )  # fmt: skip
    out, _ = await proc.communicate()
    return float(json.loads(out)["format"]["duration"])


@api.post("/transcribe")
async def transcribe(
    file: UploadFile,
    language: str = Form(DEFAULT_LANGUAGE),
    provider: str = Form(""),
) -> dict:
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language {language!r}")
    chosen = PROVIDERS.get(provider) if provider else next(iter(PROVIDERS.values()))
    if chosen is None:
        raise HTTPException(400, f"Unknown or unconfigured provider {provider!r}")
    workdir = Path(tempfile.mkdtemp(prefix="transcribe-"))
    try:
        src, flac = workdir / "upload", workdir / "audio.flac"
        await save_upload(file, src)
        await providers.ffmpeg(src, flac, "-c:a", "flac")
        src.unlink()
        duration = await duration_of(flac)
        if chosen.max_seconds and duration > chosen.max_seconds:
            raise HTTPException(
                413,
                f"{chosen.label} accepts at most {chosen.max_seconds // 60} minutes with speaker "
                f"labels; this recording is {duration / 60:.0f} minutes. Pick another model.",
            )
        segments = await chosen.transcribe(flac, workdir, language)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "provider": chosen.label,
        "duration": duration,
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
        if seg.get("start") is not None:
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
