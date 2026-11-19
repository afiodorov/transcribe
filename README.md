# Transcribe

A small web app that turns a recording (for example a 30-minute MP3) into a
transcript with speakers labelled, using
[Deepgram](https://deepgram.com)'s pre-recorded speech-to-text API with diarization.

Built for confidential conversations:

- **No training on the audio.** Every request sends `mip_opt_out=true`, which opts
  out of Deepgram's Model Improvement Program.
- **Nothing is kept.** The upload is converted in a per-request temp directory that
  is deleted when the request finishes. Transcripts exist only in the browser tab
  until you download them.
- **Password protected.** The whole site, API included, is behind HTTP basic auth.

Any audio or video format ffmpeg can read is accepted. It is converted to 16 kHz
mono FLAC (about 1 MB per minute) before upload. In the browser you can rename the
speakers ("Speaker 0" → "Therapist") and download the transcript as `.docx` or `.txt`.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
cp .env.example .env   # fill in DEEPGRAM_API_KEY, AUTH_USER, AUTH_PASSWORD
make install
make run               # http://localhost:8000
make test
```

Or in Docker: `make docker`.

## Deploy

The `Dockerfile` is all a host needs (it honours `$PORT`). Set the variables from
`.env.example` in the host's environment. Serve it over HTTPS only, because basic
auth sends the password with every request.

## License

MIT
