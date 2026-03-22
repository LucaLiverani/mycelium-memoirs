import argparse
import os
import time
from pathlib import Path

LANGUAGE = "it"


def transcribe_local(audio_path: Path, output_dir: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    model_size = "large-v3"
    compute_type = "int8"

    print(f"Loading model '{model_size}' (compute_type={compute_type})...")
    print("First run will download ~3GB model. Subsequent runs use cache.")
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=8,
    )

    print(f"Transcribing '{audio_path.name}'...")
    segments, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    result = []
    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
        print(f"  [{_fmt(segment.start)}] {result[-1]['text'][:80]}")

    print(f"Audio duration: {info.duration:.0f}s ({info.duration / 60:.1f} min)")
    return result


def transcribe_api(audio_path: Path, output_dir: Path) -> list[dict]:
    from openai import OpenAI

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("Add it to .env: OPENAI_API_KEY=sk-...")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"Transcribing '{audio_path.name}' ({file_size_mb:.1f} MB) via OpenAI API...")

    if file_size_mb > 25:
        print("Warning: File is over 25MB. Splitting not yet implemented.")
        print("Consider converting to a lower bitrate first:")
        print(f"  ffmpeg -i {audio_path} -b:a 64k {audio_path.stem}_compressed.m4a")
        raise SystemExit(1)

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=LANGUAGE,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    result = []
    for segment in response.segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
        print(f"  [{_fmt(segment.start)}] {result[-1]['text'][:80]}")

    print(f"Audio duration: {response.duration:.0f}s ({response.duration / 60:.1f} min)")
    return result


def write_outputs(segments: list[dict], stem: str, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    txt_path = output_dir / f"{stem}.txt"
    segments_path = output_dir / f"{stem}_segments.txt"
    srt_path = output_dir / f"{stem}.srt"

    with (
        open(txt_path, "w", encoding="utf-8") as f_txt,
        open(segments_path, "w", encoding="utf-8") as f_seg,
        open(srt_path, "w", encoding="utf-8") as f_srt,
    ):
        for i, seg in enumerate(segments, 1):
            f_txt.write(seg["text"] + "\n")
            f_seg.write(f"[{_fmt(seg['start'])} -> {_fmt(seg['end'])}] {seg['text']}\n")
            f_srt.write(f"{i}\n")
            f_srt.write(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n")
            f_srt.write(f"{seg['text']}\n\n")

    print(f"\nOutput files:")
    print(f"  Clean text:   {txt_path}")
    print(f"  Segments:     {segments_path}")
    print(f"  Subtitles:    {srt_path}")


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _srt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio to text")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument(
        "--api", action="store_true",
        help="Use OpenAI Whisper API instead of local model",
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory (default: output)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    output_dir = Path(args.output)

    start = time.time()

    if args.api:
        segments = transcribe_api(audio_path, output_dir)
    else:
        segments = transcribe_local(audio_path, output_dir)

    write_outputs(segments, audio_path.stem, output_dir)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
