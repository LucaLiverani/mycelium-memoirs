import argparse
import base64
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("output")

GENERATION_MODELS = {
    "story": "gpt-5.4",
    "image": "gpt-5.4",
    "video": "gpt-5.4",
}

IMAGE_MODEL = "gpt-image-1.5"
IMAGE_SIZE = "1536x1024"
IMAGE_QUALITY = "high"

VIDEO_MODEL = "sora-2-pro"
VIDEO_SIZE = "1920x1080"
VIDEO_SECONDS = "8"

INPUT_LABELS = {
    "stream": "STREAM OF CONSCIOUSNESS (during the experience)",
    "report": "EXPERIENCE REPORT (the day after)",
}


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("Add it to .env: OPENAI_API_KEY=sk-...")
        raise SystemExit(1)
    return OpenAI(api_key=api_key)


def load_prompt(gen_type: str, version: str) -> str:
    path = PROMPT_DIR / gen_type / f"{version}.md"
    if not path.exists():
        print(f"Error: Prompt template not found: {path}")
        raise SystemExit(1)
    return path.read_text(encoding="utf-8")


def build_user_content(input_paths: list[str]) -> str:
    parts = []
    for input_path in input_paths:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            raise SystemExit(1)
        text = input_file.read_text(encoding="utf-8")
        parts.append((input_file, text))

    if len(parts) == 1:
        return parts[0][1]

    # Multiple inputs: wrap each with a label
    sections = []
    label_keys = list(INPUT_LABELS.keys())
    for i, (file, text) in enumerate(parts):
        key = label_keys[i] if i < len(label_keys) else f"INPUT {i + 1}"
        label = INPUT_LABELS.get(key, key)
        sections.append(f"=== {label} ===\n{text}")
    return "\n\n".join(sections)


def generate_images(prompts: list[str], output_dir: Path, version: str):
    """Call the image generation API for each prompt and save the images."""
    client = get_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts, 1):
        print(f"  Generating image {i}/{len(prompts)}...")
        response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=IMAGE_SIZE,
            quality=IMAGE_QUALITY,
            output_format="png",
            n=1,
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        image_path = output_dir / f"{version}_{i}.png"
        image_path.write_bytes(image_bytes)
        print(f"    Saved: {image_path}")


def extend_video(client: OpenAI, video_id: str, prompt: str, seconds: int, retries: int = 3):
    """Extend a video using raw HTTP POST to avoid SDK multipart bug."""
    import httpx

    for attempt in range(1, retries + 1):
        response = httpx.post(
            "https://api.openai.com/v1/videos/extensions",
            headers={
                "Authorization": f"Bearer {client.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt,
                "seconds": str(seconds),
                "video": {"id": video_id},
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()

        error_body = response.text
        print(f"    Extension API error (attempt {attempt}/{retries}): {response.status_code}")
        print(f"    {error_body}")

        if attempt < retries:
            wait = 15 * attempt
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    print("    All retries failed.")
    raise SystemExit(1)


def generate_video(prompts: list[str], output_dir: Path, version: str, start_from: int = 1):
    """Generate a chained video using Sora 2: first clip + extensions."""
    client = get_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    last_video_id = None

    # Load video ID from previous clip if resuming
    id_file = output_dir / f"{version}_ids.txt"
    saved_ids = {}
    if id_file.exists():
        for line in id_file.read_text().strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                saved_ids[int(k)] = v

    for i, prompt in enumerate(prompts, 1):
        clip_path = output_dir / f"{version}_{i}.mp4"

        # Skip already-generated clips
        if i < start_from:
            if i in saved_ids:
                last_video_id = saved_ids[i]
                print(f"  Skipping clip {i}/{len(prompts)} (already exists, id: {last_video_id})")
            continue

        if i == 1 or last_video_id is None:
            print(f"  Generating clip {i}/{len(prompts)} ({VIDEO_SECONDS}s)...")
            video = client.videos.create(
                model=VIDEO_MODEL,
                prompt=prompt,
                size=VIDEO_SIZE,
                seconds=int(VIDEO_SECONDS),
            )
            video_id = video.id
        else:
            print(f"  Generating clip {i}/{len(prompts)} (extension, {VIDEO_SECONDS}s)...")
            data = extend_video(client, last_video_id, prompt, int(VIDEO_SECONDS))
            video_id = data["id"]

        while True:
            video = client.videos.retrieve(video_id)
            if video.status == "completed":
                break
            if video.status == "failed":
                print(f"    Clip {i} failed: {video}")
                raise SystemExit(1)
            progress = getattr(video, "progress", 0) or 0
            print(f"    Status: {video.status} ({progress}%)")
            time.sleep(10)
        print(f"    Clip {i} complete: {video_id}")

        content = client.videos.download_content(video_id, variant="video")
        clip_path.write_bytes(content.read())
        print(f"    Saved: {clip_path}")

        last_video_id = video_id

        # Save video ID for resume
        saved_ids[i] = video_id
        with open(id_file, "w") as f:
            for k in sorted(saved_ids):
                f.write(f"{k}={saved_ids[k]}\n")

    print(f"\n  All clips generated in {output_dir}/")
    print(f"  Total duration: ~{len(prompts) * int(VIDEO_SECONDS)}s")


def generate(gen_type: str, input_paths: list[str], version: str = "v1",
             start_from: int = 1, skip_text: bool = False):
    output_dir = OUTPUT_DIR / gen_type
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_text:
        # Reuse existing text prompts
        ext = ".txt"
        text_path = output_dir / f"{version}{ext}"
        if not text_path.exists():
            print(f"Error: No existing prompts at {text_path}. Run without --skip-text first.")
            raise SystemExit(1)
        result = text_path.read_text(encoding="utf-8")
        print(f"Reusing existing prompts from {text_path}")
    else:
        system_prompt = load_prompt(gen_type, version)
        user_content = build_user_content(input_paths)

        print(f"Generating {gen_type} (prompt: {version})...")
        for p in input_paths:
            print(f"  Input: {p}")
        print(f"  Model: {GENERATION_MODELS[gen_type]}")

        client = get_client()
        response = client.chat.completions.create(
            model=GENERATION_MODELS[gen_type],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        result = response.choices[0].message.content

        print(f"  Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")

        if gen_type == "story" and "\n---\n" in result:
            parts = result.split("\n---\n", 1)
            it_path = output_dir / f"{version}_it.md"
            en_path = output_dir / f"{version}_en.md"
            it_path.write_text(parts[0].strip(), encoding="utf-8")
            en_path.write_text(parts[1].strip(), encoding="utf-8")
            print(f"  Output (IT): {it_path}")
            print(f"  Output (EN): {en_path}")
        else:
            ext = ".md" if gen_type == "story" else ".txt"
            output_path = output_dir / f"{version}{ext}"
            output_path.write_text(result, encoding="utf-8")
            print(f"  Output: {output_path}")

    print()
    print(result)

    # For image type: also generate actual images from the prompts
    if gen_type == "image":
        prompts = [p.strip() for p in result.strip().split("\n\n") if p.strip()]
        print(f"\nGenerating {len(prompts)} images with {IMAGE_MODEL}...")
        generate_images(prompts, output_dir, version)

    # For video type: generate video clips with Sora 2
    if gen_type == "video":
        prompts = [p.strip() for p in result.strip().split("\n\n") if p.strip()]
        print(f"\nGenerating {len(prompts)} video clips with {VIDEO_MODEL}...")
        generate_video(prompts, output_dir, version, start_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate creative content from transcripts and stories",
    )
    parser.add_argument(
        "type",
        choices=["story", "image", "video"],
        help="Type of content to generate",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Path to input file(s). For story: pass stream-of-consciousness first, then experience report.",
    )
    parser.add_argument(
        "--prompt-version", default="v1",
        help="Prompt template version (default: v1)",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="For video: resume from clip N (skips earlier clips)",
    )
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Skip text generation and reuse existing prompts from output/",
    )
    args = parser.parse_args()

    generate(args.type, args.input, args.prompt_version, args.start_from, args.skip_text)
