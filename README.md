# mycelium-memoirs

A creative pipeline that transforms spoken experience reports into written stories, image prompts, and video prompts. Records personal experiences with psychedelic mushrooms and turns them into art — from raw Italian audio to diary entries, visual art directions, and cinematic sequences.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

## Pipeline

```
Audio → Transcript → Story → Images (DALL-E) / Video (Sora 2)
```

### 1. Transcribe audio

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

Output goes to `output/`:
- `*.txt` — clean text
- `*_segments.txt` — timestamped segments
- `*.srt` — subtitles

### 2. Generate story

Combines multiple transcripts into a personal blog post. Pass the stream of consciousness (recorded during) first, then the experience report (recorded after):
```bash
python3 generate.py story \
  output/experince-stream-of-consciousness.txt \
  output/experience-report.txt
```

Output is automatically split into separate Italian and English files:
- `output/story/v1_it.md` — Italian version
- `output/story/v1_en.md` — English version

You can also use a single input:
```bash
python3 generate.py story output/experience-report.txt
```

### 3. Generate images

Generates text prompts from the story, then creates actual images via GPT Image 1.5:
```bash
python3 generate.py image output/story/v1_en.md
```

Output: `output/image/v1.txt` (prompts) + `output/image/v1_1.png`, `v1_2.png`, etc.

### 4. Generate video

Generates clip prompts from the story, then creates a ~30s chained video via Sora 2 Pro:
```bash
python3 generate.py video output/story/v1_en.md
```

Output: `output/video/v1.txt` (prompts) + `output/video/v1_1.mp4`, `v1_2.mp4`, etc. (chained via Sora extensions)

### Options

```bash
# Use a different prompt version
python3 generate.py story output/experience-report.txt --prompt-version v2

# Reuse existing text prompts (skip LLM call), useful when resuming
python3 generate.py video output/story/v1_en.md --skip-text

# Resume video generation from a specific clip
python3 generate.py video output/story/v1_en.md --skip-text --start-from 3
```

## Project structure

```
prompts/          # Prompt templates (versioned, human-editable)
  story/v1.md
  image/v1.md
  video/v1.md
output/           # Generated content
  *.txt, *.srt    # Transcriptions
  story/v1_it.md  # Story (Italian)
  story/v1_en.md  # Story (English)
  image/v1.txt    # Image prompts
  image/v1_*.png  # Generated images (GPT Image 1.5)
  video/v1.txt    # Video prompts
  video/v1_*.mp4  # Generated clips (Sora 2 Pro)
```

Prompt templates and outputs are plain text — easy to copy-paste into any platform. Version with v1, v2, etc. to experiment.
