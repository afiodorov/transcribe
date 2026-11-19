import io
import os
import subprocess

os.environ["DEEPGRAM_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY"] = "gemini-key"
os.environ["OPENAI_API_KEY"] = "openai-key"
TOKEN = "t" * 32
os.environ["ACCESS_TOKEN"] = TOKEN

import asyncio  # noqa: E402

import docx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main, providers  # noqa: E402

client = TestClient(main.app)
authed = TestClient(main.app, cookies={"access": TOKEN})

DEEPGRAM_RESULT = {
    "metadata": {"duration": 12.5},
    "results": {
        "utterances": [
            {"speaker": 0, "start": 0.0, "end": 2.0, "transcript": "How was your week?"},
            {"speaker": 1, "start": 2.5, "end": 4.0, "transcript": "Better."},
            {"speaker": 1, "start": 4.1, "end": 6.0, "transcript": "I slept more."},
            {"speaker": 0, "start": 6.5, "end": 8.0, "transcript": "  "},
            {"speaker": 0, "start": 8.0, "end": 9.0, "transcript": "Good."},
        ]
    },
}


def tone_mp3() -> bytes:
    return subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-f",
            "mp3",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout


@pytest.mark.parametrize("cookie", [None, "wrong", TOKEN[:-1]])
def test_every_route_requires_the_access_cookie(cookie):
    c = TestClient(main.app, cookies={"access": cookie} if cookie else None)
    for method, path in [
        ("get", "/"),
        ("get", "/api/config"),
        ("post", "/api/transcribe"),
        ("post", "/api/docx"),
    ]:
        r = getattr(c, method)(path)
        assert r.status_code == 401, path
    assert "link you were given" in c.get("/").text
    assert "<script" not in c.get("/").text


def test_link_sets_a_long_lived_cookie_and_redirects_to_a_clean_url():
    c = TestClient(main.app, base_url="https://testserver")
    r = c.get(f"/s/{TOKEN}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    for attr in ["access=" + TOKEN, "httponly", "secure", "samesite=lax", "max-age=31536000"]:
        assert attr in cookie
    assert c.get("/").status_code == 200, "the cookie alone must now be enough"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_wrong_link_sets_no_cookie():
    r = client.get("/s/" + "x" * 32, follow_redirects=False)
    assert r.status_code == 401
    assert "set-cookie" not in r.headers


def test_index_served_with_cookie():
    r = authed.get("/")
    assert r.status_code == 200
    assert "Transcribe" in r.text


GEMINI_RESULT = {
    "status": "completed",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "…", "annotations": [
        {"type": "word_info", "text": "Hola,", "speaker": "spk_2"},
        {"type": "word_info", "text": "¿qué tal?", "speaker": "spk_2"},
        {"type": "word_info", "text": "Bien.", "speaker": "spk_1"},
        {"type": "word_info", "text": "Vale.", "speaker": "spk_2", "start_offset": "3.5s"},
    ]}]}],
}  # fmt: skip

OPENAI_RESULT = {
    "segments": [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": " How are you?"},
        {"speaker": "B", "start": 1.2, "end": 2.0, "text": "Fine."},
        {"speaker": "B", "start": 2.1, "end": 3.0, "text": "Thanks."},
    ]
}


class FakeClient:
    """Stands in for httpx.AsyncClient and records the one request made."""

    calls: list = []
    reply: dict = {}

    def __init__(self, **_):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def post(self, url, **kwargs):
        FakeClient.calls.append((url, kwargs))
        return providers.httpx.Response(200, json=FakeClient.reply)


@pytest.fixture
def fake_http(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", FakeClient)
    return FakeClient


@pytest.fixture
def flac(tmp_path):
    path = tmp_path / "audio.flac"
    asyncio.run(providers.ffmpeg(write(tmp_path / "in.mp3", tone_mp3()), path, "-c:a", "flac"))
    return path


def write(path, data):
    path.write_bytes(data)
    return path


def test_deepgram_merges_same_speaker_turns_and_drops_blank_ones():
    segs = providers.Deepgram.parse(DEEPGRAM_RESULT)
    assert [(s.speaker, s.text) for s in segs] == [
        (0, "How was your week?"),
        (1, "Better. I slept more."),
        (0, "Good."),
    ]
    assert (segs[1].start, segs[1].end) == (2.5, 6.0)


def test_deepgram_request_opts_out_of_training_and_uses_v2_speakers(fake_http, flac):
    fake_http.reply = DEEPGRAM_RESULT
    asyncio.run(providers.Deepgram().transcribe(flac, flac.parent, "en"))
    ((url, kw),) = fake_http.calls
    p = kw["params"]
    assert p["mip_opt_out"] == "true"
    assert p["diarize_model"] == "latest"
    assert "diarize" not in p
    assert p["utterances"] == "true"
    assert kw["headers"]["Authorization"] == "Token test-key"


def test_gemini_numbers_speakers_by_first_appearance_and_groups_words():
    segs = providers.Gemini.parse(GEMINI_RESULT)
    assert [(s.speaker, s.text) for s in segs] == [
        (0, "Hola, ¿qué tal?"),
        (1, "Bien."),
        (0, "Vale."),
    ]
    assert segs[0].start is None and segs[2].start == 3.5


def test_gemini_request_is_inline_unstored_verbatim_diarized(fake_http, flac):
    fake_http.reply = GEMINI_RESULT
    asyncio.run(providers.Gemini().transcribe(flac, flac.parent, "es"))
    ((url, kw),) = fake_http.calls
    body = kw["json"]
    assert url.endswith("/v1beta/interactions")
    assert body["store"] is False, "Gemini keeps requests 55 days unless store=false"
    assert body["model"] == "models/gemini-3.5-transcribe"
    audio = body["input"][0]
    assert audio["type"] == "audio" and audio["mime_type"] == "audio/flac"
    assert providers.base64.b64decode(audio["data"])[:4] == b"fLaC"
    tc = body["generation_config"]["transcription_config"]
    assert tc["mode"] == {"type": "verbatim", "diarization_mode": "speaker"}
    assert tc["language_codes"] == ["es-ES"]
    assert kw["headers"]["x-goog-api-key"] == "gemini-key"


def test_gemini_mixed_languages_means_auto_detect(fake_http, flac):
    fake_http.reply = GEMINI_RESULT
    asyncio.run(providers.Gemini().transcribe(flac, flac.parent, "multi"))
    assert fake_http.calls[0][1]["json"]["generation_config"]["transcription_config"][
        "language_codes"] == []  # fmt: skip


def test_openai_sends_small_opus_to_the_diarize_model(fake_http, flac):
    fake_http.reply = OPENAI_RESULT
    segs = asyncio.run(providers.OpenAI().transcribe(flac, flac.parent, "es"))
    ((url, kw),) = fake_http.calls
    assert url.endswith("/v1/audio/transcriptions")
    assert kw["data"] == {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
        "language": "es",
    }
    name, data, mime = kw["files"]["file"]
    assert data[:4] == b"OggS" and mime == "audio/ogg"
    assert kw["headers"]["Authorization"] == "Bearer openai-key"
    assert [(s.speaker, s.text) for s in segs] == [(0, "How are you?"), (1, "Fine. Thanks.")]


def test_config_lists_configured_providers():
    ids = [p["id"] for p in authed.get("/api/config").json()["providers"]]
    assert ids == ["deepgram", "gemini", "openai"]


@pytest.mark.parametrize("provider", ["deepgram", "gemini", "openai"])
def test_transcribe_routes_to_the_chosen_provider_and_cleans_up(monkeypatch, provider):
    seen = {}

    async def fake(flac, workdir, language):
        seen.update(header=flac.read_bytes()[:4], language=language, dir=workdir)
        return [providers.Segment(0, 0.0, 1.0, "Hi."), providers.Segment(1, 1.0, 2.0, "Hey.")]

    monkeypatch.setattr(main.PROVIDERS[provider], "transcribe", fake)
    r = authed.post(
        "/api/transcribe",
        data={"language": "es", "provider": provider},
        files={"file": ("session.mp3", tone_mp3(), "audio/mpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == main.PROVIDERS[provider].label
    assert 1.9 < body["duration"] < 2.2
    assert body["speakers"] == [0, 1]
    assert seen["header"] == b"fLaC"
    assert seen["language"] == "es"
    assert not seen["dir"].exists(), "temp dir with the audio must be deleted"


def test_too_long_for_gemini_is_refused_before_sending(monkeypatch):
    called = []
    monkeypatch.setattr(main.PROVIDERS["gemini"], "max_seconds", 1)
    monkeypatch.setattr(main.PROVIDERS["gemini"], "transcribe", lambda *a: called.append(a))
    r = authed.post(
        "/api/transcribe",
        data={"provider": "gemini"},
        files={"file": ("long.mp3", tone_mp3(), "audio/mpeg")},
    )
    assert r.status_code == 413
    assert "silently cut" in r.json()["detail"]
    assert not called


def test_unknown_provider_is_rejected():
    r = authed.post(
        "/api/transcribe",
        data={"provider": "whisper"},
        files={"file": ("a.mp3", b"x", "audio/mpeg")},
    )
    assert r.status_code == 400


def test_non_audio_upload_is_rejected():
    r = authed.post(
        "/api/transcribe",
        files={"file": ("notes.txt", b"not audio at all", "text/plain")},
    )
    assert r.status_code == 422
    assert "Could not read" in r.json()["detail"]


def test_unknown_language_is_rejected():
    r = authed.post(
        "/api/transcribe",
        data={"language": "xx"},
        files={"file": ("a.mp3", b"x", "audio/mpeg")},
    )
    assert r.status_code == 400


def test_docx_uses_speaker_names_and_timestamps():
    segs = [
        {"speaker": 0, "start": 65, "end": 70, "text": "Hello."},
        {"speaker": 1, "start": 3725, "end": 3730, "text": "Hi."},
    ]
    r = authed.post(
        "/api/docx",
        json={"title": "Session", "names": {"0": "Therapist"}, "segments": segs},
    )
    assert r.status_code == 200
    paras = [p.text for p in docx.Document(io.BytesIO(r.content)).paragraphs]
    assert paras == ["Session", "[00:01:05] Therapist: Hello.", "[01:02:05] Speaker 1: Hi."]


def test_docx_without_timestamps_omits_them():
    segs = [{"speaker": 0, "start": None, "end": None, "text": "Hola."}]
    r = authed.post("/api/docx", json={"title": "S", "names": {}, "segments": segs})
    paras = [p.text for p in docx.Document(io.BytesIO(r.content)).paragraphs]
    assert paras == ["S", "Speaker 0: Hola."]
