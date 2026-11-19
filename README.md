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
- **Private link.** Access is a shareable link, `/s/<ACCESS_TOKEN>`. Opening it
  stores the token in an HttpOnly cookie for a year and redirects to `/`, so the
  token does not stay in the address bar or history. Without the cookie the site
  shows nothing. Rotate `ACCESS_TOKEN` to revoke access.

Any audio or video format ffmpeg can read is accepted. It is converted to 16 kHz
mono FLAC (about 1 MB per minute) before upload. In the browser you can rename the
speakers ("Speaker 0" → "Therapist") and download the transcript as `.docx` or `.txt`.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
cp .env.example .env   # fill in DEEPGRAM_API_KEY and ACCESS_TOKEN
make install
make run
make link              # prints the link to open
make test
```

Or in Docker: `make docker`.

## Deploy

The `Dockerfile` is all a host needs (it honours `$PORT`). Set the variables from
`.env.example` in the host's environment. Serve it over HTTPS only: the link and the cookie are the credential.

## License

MIT
