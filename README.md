# Transcribe

A small web app that turns a recording (for example a 30-minute MP3) into a
transcript with speakers labelled. Pick a model per upload:

| Model | Notes |
| --- | --- |
| Deepgram Nova-3 | `diarize_model=latest` (v2 speaker model). `mip_opt_out=true` opts out of Deepgram's Model Improvement Program, so audio is not kept for training. |
| Gemini 3.5 Transcribe | Audio sent inline (no Files API copy) with `store: false` (the Interactions API otherwise keeps requests 55 days). With speaker labels Gemini handles at most 30 minutes and silently truncates the rest, so longer recordings are refused. Google trains on free-tier data: use a key from a project with billing on. |
| OpenAI GPT-4o Transcribe Diarize | OpenAI's only speaker-labelling model; deprecated, shuts down 2027-02-26. Sent as 32 kbps mono Opus to fit the 25 MB limit. API data is not used for training. |

Only models whose API key is set appear in the dropdown.

Built for confidential conversations:

- **Nothing is kept.** The upload is converted in a per-request temp directory that
  is deleted when the request finishes. Transcripts exist only in the browser tab
  until you download them.
- **Private link.** Access is a shareable link, `/s/<ACCESS_TOKEN>`. Opening it
  stores the token in an HttpOnly cookie for a year and redirects to `/`, so the
  token does not stay in the address bar or history. Without the cookie the site
  shows nothing. Rotate `ACCESS_TOKEN` to revoke access.

Any audio or video format ffmpeg can read is accepted; it is converted to 16 kHz
mono FLAC first. In the browser you can rename the speakers ("Speaker 0" →
"Therapist") and download the transcript as `.docx` or `.txt`.

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
