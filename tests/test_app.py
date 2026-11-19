import io
import os
import subprocess

os.environ.setdefault("DEEPGRAM_API_KEY", "test-key")
TOKEN = "t" * 32
os.environ["ACCESS_TOKEN"] = TOKEN

import docx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402

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


def test_consecutive_same_speaker_turns_merge_and_blank_ones_drop():
    segs = main.to_segments(DEEPGRAM_RESULT)
    assert [(s.speaker, s.text) for s in segs] == [
        (0, "How was your week?"),
        (1, "Better. I slept more."),
        (0, "Good."),
    ]
    assert (segs[1].start, segs[1].end) == (2.5, 6.0)


def test_transcribe_converts_to_flac_sends_privacy_params_and_cleans_up(monkeypatch, tmp_path):
    seen = {}

    async def fake_deepgram(audio, language):
        seen["header"] = audio.read_bytes()[:4]
        seen["language"] = language
        seen["dir"] = audio.parent
        return DEEPGRAM_RESULT

    monkeypatch.setattr(main, "call_deepgram", fake_deepgram)
    r = authed.post(
        "/api/transcribe",
        data={"language": "es"},
        files={"file": ("session.mp3", tone_mp3(), "audio/mpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duration"] == 12.5
    assert body["speakers"] == [0, 1]
    assert len(body["segments"]) == 3
    assert seen["header"] == b"fLaC"
    assert seen["language"] == "es"
    assert not seen["dir"].exists(), "temp dir with the audio must be deleted"


def test_deepgram_request_opts_out_of_training_and_diarizes(monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, url, params, content, headers):
            captured.update(url=url, params=params, headers=headers)
            return main.httpx.Response(200, json=DEEPGRAM_RESULT)

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"fLaC")
    import asyncio

    asyncio.run(main.call_deepgram(audio, "en"))
    p = captured["params"]
    assert p["mip_opt_out"] == "true"
    assert p["diarize_model"] == "latest"
    assert "diarize" not in p
    assert p["utterances"] == "true"
    assert captured["headers"]["Authorization"] == "Token test-key"


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
