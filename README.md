# mycelium-memoirs
A personal archive at the intersection of consciousness, fungi, and art.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Transcribe audio

Place your audio file in the `audio/` directory.

**Local (free, slower ~1-2x realtime):**
```bash
python3 transcribe.py audio/experience-report.m4a
```
First run downloads the Whisper `large-v3` model (~3GB, cached for future runs).

**OpenAI API (fast, ~$0.006/min):**
```bash
python3 transcribe.py audio/experience-report.m4a --api
```
Reads `OPENAI_API_KEY` from `.env`.

Output goes to `output/`:
- `*.txt` — clean text
- `*_segments.txt` — timestamped segments
- `*.srt` — subtitles
