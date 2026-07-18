#!/usr/bin/env python3
"""Generate a cached character-motion GIF for the current Spotify top track.

The script has two GitHub Actions-friendly phases:

    python generate_top_image.py metadata --output .animation/metadata.json
    python generate_top_image.py render \
        --metadata .animation/metadata.json \
        --cache-dir .animation-cache \
        --site-dir _site \
        --soft-fail

The metadata phase only calls Spotify and emits a deterministic asset key. The
render phase reuses the Actions cache when possible. Otherwise it asks GPT
Image to create one subtly moved keyframe from the real album cover, estimates
the local motion, and applies that motion to the original cover pixels.
"""

import argparse
import base64
import hashlib
import html
import json
import math
import os
import shutil
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps

import refresh_widget


OPENAI_IMAGE_EDIT_URL = "https://api.openai.com/v1/images/edits"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_IMAGE_QUALITY = "medium"
PROMPT_VERSION = "album-character-articulation-v2"
RENDER_VERSION = "optical-flow-safe-canvas-v3"
MAX_GIF_BYTES = 2 * 1024 * 1024
MAX_SOURCE_IMAGE_BYTES = 20 * 1024 * 1024
SAFE_CONTENT_SCALE = 0.85
SAFE_BACKGROUND_BRIGHTNESS = 0.58
GIF_PROFILES = (
    (384, 20, 96, 100),
    (320, 18, 96, 100),
    (256, 16, 64, 125),
)


class AnimationError(RuntimeError):
    """Expected generation or rendering failure."""


def write_github_output(name, value):
    """Write a simple one-line output when running in GitHub Actions."""
    value = str(value)
    if "\n" in name or "\n" in value:
        raise ValueError("GitHub outputs must be single-line values")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write(f"{name}={value}\n")
    print(f"{name}={value}")


def _load_overrides():
    path = Path(__file__).with_name("overrides.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def load_spotify_config():
    """Load only the Spotify settings needed by the metadata phase."""
    env_keys = {
        "spotify_client_id": "SPOTIFY_CLIENT_ID",
        "spotify_client_secret": "SPOTIFY_CLIENT_SECRET",
        "spotify_refresh_token": "SPOTIFY_REFRESH_TOKEN",
    }
    cfg = {key: os.environ.get(env, "").strip() for key, env in env_keys.items()}

    # Local runs may use the ignored config.json, while Actions uses secrets.
    config_path = Path(__file__).with_name("config.json")
    if config_path.exists():
        local = json.loads(config_path.read_text(encoding="utf-8"))
        for key in env_keys:
            if not cfg[key]:
                cfg[key] = str(local.get(key, "")).strip()

    missing = [key for key in env_keys if not cfg[key]]
    if missing:
        raise AnimationError("Spotify settings are missing: " + ", ".join(missing))

    cfg["time_range"] = os.environ.get("TIME_RANGE", "").strip() or "short_term"
    cfg["name_overrides"] = _load_overrides()
    return cfg


def _clean_metadata(value, limit=180):
    value = " ".join(str(value or "").split())
    return value[:limit]


def build_prompt(metadata):
    """Describe one subtle, character-focused alternate keyframe."""
    return (
        "Inspect the supplied square album cover and edit it into exactly one peak "
        "motion keyframe for a subtle Live2D-style character loop. Articulate the main "
        "illustrated character instead of translating the whole character or cover as "
        "one rigid layer. When eyes are visible, close them gently in a clearly readable "
        "natural blink while preserving the face and expression. Add restrained "
        "follow-through to loose hair, fabric, and accessories, plus a small natural "
        "change to any prominently posed arm or hand. Keep every movement local and "
        "roughly one to three percent of the canvas. If no character is present, move "
        "only one appropriate illustrated foreground element. Preserve the exact "
        "illustration style, character identity, composition, camera, crop, colors, "
        "background, typography, logos, and every object. Do not create a new scene or "
        "add, remove, replace, or redesign anything. Keep all text unchanged and in "
        "exactly the same position. The camera must remain completely locked: no pan, "
        "zoom, rotation, reframing, lighting change, color shift, transition, border, "
        "caption, or new text."
    )


def _clean_publication_id(value):
    value = str(value or "").strip()
    if not value:
        return ""
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "-_")
    if cleaned != value:
        raise AnimationError("Publication ID contains unsupported characters")
    return cleaned[:80]


def build_metadata(track, model=None, quality=None, publication_id=None):
    model = model or os.environ.get("OPENAI_IMAGE_MODEL", "").strip()
    model = model or DEFAULT_IMAGE_MODEL
    quality = quality or os.environ.get("OPENAI_IMAGE_QUALITY", "").strip()
    quality = quality or DEFAULT_IMAGE_QUALITY

    metadata = {
        "track_id": _clean_metadata(track.get("id"), 100),
        "name": _clean_metadata(track.get("name")),
        "artist": _clean_metadata(track.get("artist")),
        "source_image_url": str(track.get("art") or "").strip(),
        "model": model,
        "quality": quality,
        "prompt_version": PROMPT_VERSION,
        "render_version": RENDER_VERSION,
    }
    identity = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    asset_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    metadata["asset_key"] = asset_key
    metadata["filename"] = f"top-track-{asset_key}.gif"
    publication_id = _clean_publication_id(publication_id)
    publication_suffix = f"-{publication_id}" if publication_id else ""
    metadata["site_path"] = (
        f"animations/top-track-{asset_key}{publication_suffix}.gif"
    )
    metadata["relative_path"] = metadata["site_path"]
    return metadata


def prepare_metadata(output_path, publication_id=None):
    cfg = load_spotify_config()
    access_token = refresh_widget.get_spotify_access_token(cfg)
    tracks = refresh_widget.get_top_tracks(
        access_token,
        cfg["time_range"],
        limit=1,
        overrides=cfg.get("name_overrides"),
    )
    if not tracks:
        raise AnimationError("Spotify returned no top tracks")

    metadata = build_metadata(tracks[0], publication_id=publication_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_github_output("asset_key", metadata["asset_key"])
    write_github_output("relative_path", metadata["relative_path"])
    print(f"Animation target: {metadata['name']} — {metadata['artist']}")
    return metadata


def download_source_image(metadata, max_bytes=MAX_SOURCE_IMAGE_BYTES):
    """Download and validate the top track album cover from Spotify."""
    image_url = str(metadata.get("source_image_url") or "").strip()
    if not image_url.startswith("https://"):
        raise AnimationError("Spotify cover URL is missing or is not HTTPS")
    try:
        response = requests.get(image_url, timeout=30)
    except requests.RequestException as exc:
        raise AnimationError(f"Spotify cover download failed: {exc}") from exc

    if response.status_code != 200:
        raise AnimationError(
            f"Spotify cover download failed ({response.status_code})"
        )
    content_type = str(response.headers.get("Content-Type") or "").lower()
    image_bytes = response.content
    if not content_type.startswith("image/"):
        raise AnimationError(
            f"Spotify cover has unexpected type: {content_type or 'unknown'}"
        )
    if not image_bytes or len(image_bytes) > max_bytes:
        raise AnimationError("Spotify cover is empty or exceeds the size limit")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise AnimationError("Spotify returned invalid cover image data") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width != image.height:
        raise AnimationError(
            "Spotify cover must be square to animate without cropping"
        )
    return image


def generate_motion_keyframe(metadata, source_image):
    """Ask GPT Image for one small semantic movement of the source cover."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AnimationError("OPENAI_API_KEY is not configured")

    prepared = ImageOps.fit(
        source_image.convert("RGB"),
        (1024, 1024),
        method=Image.Resampling.LANCZOS,
    )
    source_buffer = BytesIO()
    prepared.save(source_buffer, format="PNG")
    payload = {
        "model": metadata.get("model") or DEFAULT_IMAGE_MODEL,
        "prompt": build_prompt(metadata),
        "n": "1",
        "size": "1024x1024",
        "quality": metadata.get("quality") or DEFAULT_IMAGE_QUALITY,
        "output_format": "png",
    }
    try:
        response = requests.post(
            OPENAI_IMAGE_EDIT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=payload,
            files={
                "image[]": (
                    "album-cover.png",
                    source_buffer.getvalue(),
                    "image/png",
                )
            },
            timeout=300,
        )
    except requests.RequestException as exc:
        raise AnimationError(f"OpenAI image edit request failed: {exc}") from exc

    if response.status_code != 200:
        message = response.text[:500]
        try:
            message = response.json().get("error", {}).get("message") or message
        except Exception:
            pass
        raise AnimationError(
            f"OpenAI image edit failed ({response.status_code}): {message}"
        )

    try:
        encoded = response.json()["data"][0]["b64_json"]
        image_bytes = base64.b64decode(encoded, validate=True)
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise AnimationError("OpenAI returned invalid edited image data") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def _fit_rgb_array(image, size):
    fitted = ImageOps.fit(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    return np.asarray(fitted, dtype=np.uint8)


def _safe_content_size(size):
    """Return an evenly centered cover size with a crop-safe outer margin."""
    if size < 8:
        raise ValueError("Animation canvas must be at least 8 pixels")
    content_size = int(round(size * SAFE_CONTENT_SCALE))
    content_size = max(2, min(size - 2, content_size))
    if (size - content_size) % 2:
        content_size -= 1
    return content_size


def _safe_canvas_background(source_image, size):
    """Build a static dark blurred backdrop from the complete album cover."""
    fitted = ImageOps.fit(
        source_image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    blurred = fitted.filter(
        ImageFilter.GaussianBlur(radius=max(4.0, size * 0.055))
    )
    background = np.asarray(blurred, dtype=np.float32)
    return np.clip(
        background * SAFE_BACKGROUND_BRIGHTNESS, 0, 255
    ).astype(np.uint8)


def _composite_safe_canvas(background, content):
    """Center the full cover motion frame over its static safe-area backdrop."""
    canvas = background.copy()
    height, width = canvas.shape[:2]
    content_height, content_width = content.shape[:2]
    if content_height > height or content_width > width:
        raise ValueError("Cover content does not fit inside the animation canvas")
    top = (height - content_height) // 2
    left = (width - content_width) // 2
    canvas[top : top + content_height, left : left + content_width] = content
    return canvas


def _align_keyframe(source, keyframe):
    """Remove unintended camera drift before measuring local character motion."""
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    keyframe_gray = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            source_gray,
            keyframe_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        raise AnimationError(
            "GPT Image keyframe could not be aligned with the album cover"
        ) from exc

    if not np.isfinite(float(correlation)) or not np.isfinite(warp).all():
        raise AnimationError(
            "GPT Image returned an invalid cover alignment"
        )
    linear_transform = warp[:, :2]
    determinant = float(np.linalg.det(linear_transform))
    singular_values = np.linalg.svd(linear_transform, compute_uv=False)
    rotation_degrees = abs(
        math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0])))
    )
    height, width = source.shape[:2]
    translation_ratio = max(
        abs(float(warp[0, 2])) / max(1, width),
        abs(float(warp[1, 2])) / max(1, height),
    )
    if (
        float(correlation) < 0.85
        or determinant <= 0.0
        or float(singular_values.min()) < 0.97
        or float(singular_values.max()) > 1.03
        or rotation_degrees > 3.0
        or translation_ratio > 0.04
    ):
        raise AnimationError(
            "GPT Image changed the cover framing too much to animate safely"
        )

    print(
        "Keyframe alignment: "
        f"correlation={float(correlation):.3f}, "
        f"translation={translation_ratio:.2%}"
    )
    return cv2.warpAffine(
        keyframe,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _character_motion_flow(source, keyframe, max_motion=12.0):
    """Estimate a smooth, bounded local deformation from one semantic keyframe."""
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    keyframe_gray = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)
    difference = np.mean(
        np.abs(source.astype(np.float32) - keyframe.astype(np.float32)),
        axis=2,
    )
    changed_fraction = float(np.mean(difference > 28.0))
    if float(np.median(difference)) > 18.0 or changed_fraction > 0.55:
        raise AnimationError(
            "GPT Image redrew too much of the album cover to animate safely"
        )

    flow = cv2.calcOpticalFlowFarneback(
        source_gray,
        keyframe_gray,
        None,
        0.5,
        4,
        25,
        4,
        7,
        1.5,
        0,
    )

    # Remove any residual whole-image movement, then keep only areas that the
    # edited keyframe meaningfully changed. This protects static cover text.
    flow -= np.median(flow.reshape(-1, 2), axis=0).astype(np.float32)
    motion_mask = np.clip((difference - 4.0) / 36.0, 0.0, 1.0)
    motion_mask = cv2.GaussianBlur(motion_mask, (0, 0), sigmaX=5.0)

    height, width = motion_mask.shape
    edge_y = np.minimum(np.arange(height), np.arange(height)[::-1])
    edge_x = np.minimum(np.arange(width), np.arange(width)[::-1])
    edge_fade = np.minimum(edge_y[:, None], edge_x[None, :])
    edge_fade = np.clip(
        edge_fade / max(8.0, min(height, width) * 0.04), 0.0, 1.0
    )
    motion_mask *= edge_fade.astype(np.float32)

    magnitude = np.linalg.norm(flow, axis=2)
    limiter = np.minimum(1.0, max_motion / np.maximum(magnitude, 1e-6))
    flow *= limiter[..., None].astype(np.float32)
    flow *= motion_mask[..., None].astype(np.float32)
    flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=1.4)

    usable_magnitude = np.linalg.norm(flow, axis=2)
    active_fraction = float(np.mean(usable_magnitude > 0.25))
    top_count = max(1, usable_magnitude.size // 100)
    top_motion_mean = float(
        np.mean(np.partition(usable_magnitude.ravel(), -top_count)[-top_count:])
    )
    if active_fraction < 0.001 or top_motion_mean < 0.25:
        raise AnimationError("GPT Image keyframe contained too little usable motion")
    if active_fraction > 0.40:
        raise AnimationError(
            "GPT Image keyframe moved too much of the album cover"
        )
    print(
        "Motion guide: "
        f"active_area={active_fraction:.2%}, "
        f"strongest_motion={top_motion_mean:.2f}px"
    )
    return flow


def _animation_frames(source_image, keyframe_image, size, frame_count, colors):
    content_size = _safe_content_size(size)
    source = _fit_rgb_array(source_image, content_size)
    keyframe = _fit_rgb_array(keyframe_image, content_size)
    keyframe = _align_keyframe(source, keyframe)
    flow = _character_motion_flow(
        source,
        keyframe,
        max_motion=max(5.0, content_size * 0.035),
    )

    background = _safe_canvas_background(source_image, size)
    palette_source = _composite_safe_canvas(background, source)
    palette_seed = Image.fromarray(palette_source).quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(content_size, dtype=np.float32),
        np.arange(content_size, dtype=np.float32),
    )
    print(
        f"Safe canvas: cover={content_size}x{content_size} "
        f"inside {size}x{size} ({content_size / size:.0%})"
    )

    frames = []
    for index in range(frame_count):
        phase = (2.0 * math.pi * index) / frame_count
        amount = 0.5 - 0.5 * math.cos(phase)
        map_x = grid_x - flow[..., 0] * amount
        map_y = grid_y - flow[..., 1] * amount
        motion_frame = cv2.remap(
            source,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        frame_array = _composite_safe_canvas(background, motion_frame)
        frame = Image.fromarray(frame_array)
        frames.append(
            frame.quantize(palette=palette_seed, dither=Image.Dither.NONE)
        )
    return frames


def is_valid_animated_gif(path, max_bytes=MAX_GIF_BYTES):
    path = Path(path)
    if (
        not path.exists()
        or path.stat().st_size == 0
        or path.stat().st_size > max_bytes
    ):
        return False
    try:
        with Image.open(path) as image:
            return (
                image.format == "GIF"
                and bool(getattr(image, "is_animated", False))
                and int(getattr(image, "n_frames", 1)) > 1
                and image.width == image.height
            )
    except Exception:
        return False


def create_looping_gif(
    source_image,
    keyframe_image,
    output_path,
    profiles=None,
    max_bytes=MAX_GIF_BYTES,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles = profiles or GIF_PROFILES
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    try:
        for size, frame_count, colors, duration_ms in profiles:
            frames = _animation_frames(
                source_image, keyframe_image, size, frame_count, colors
            )
            frames[0].save(
                temporary_path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
                # Frame 0 initializes the opaque canvas. Retaining each frame lets
                # Pillow encode only the moving center instead of the static backdrop.
                disposal=1,
            )
            if is_valid_animated_gif(temporary_path, max_bytes=max_bytes):
                temporary_path.replace(output_path)
                print(
                    f"GIF created: {size}x{size}, {frame_count} frames, "
                    f"{output_path.stat().st_size} bytes"
                )
                return output_path
            temporary_path.unlink(missing_ok=True)
    finally:
        temporary_path.unlink(missing_ok=True)

    raise AnimationError("Could not produce an animated GIF under the size limit")


def _write_site_index(site_dir, metadata):
    site_dir = Path(site_dir)
    title = html.escape(metadata.get("name", "Top Track"))
    artist = html.escape(metadata.get("artist", ""))
    relative_path = html.escape(metadata["relative_path"], quote=True)
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spotify On Repeat animation</title>
  <style>
    html, body {{ min-height: 100%; margin: 0; background: #0d0d0d; color: #fff; }}
    body {{ display: grid; place-items: center; font: 16px system-ui, sans-serif; }}
    main {{ width: min(86vw, 480px); text-align: center; }}
    img {{ display: block; width: 100%; border-radius: 24px; }}
    p {{ color: #aaa; }}
  </style>
</head>
<body>
  <main>
    <img src="{relative_path}" alt="Generated animation for {title}">
    <h1>{title}</h1>
    <p>{artist}</p>
  </main>
</body>
</html>
"""
    (site_dir / "index.html").write_text(index, encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")


def render_animation(metadata_path, cache_dir, site_dir):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    required = (
        "asset_key",
        "filename",
        "site_path",
        "relative_path",
        "source_image_url",
        "name",
        "artist",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise AnimationError("Animation metadata is missing: " + ", ".join(missing))

    cache_dir = Path(cache_dir)
    site_dir = Path(site_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / metadata["filename"]
    cache_hit = is_valid_animated_gif(cache_file)

    if cache_hit:
        print(f"Reusing cached animation: {cache_file}")
    else:
        cache_file.unlink(missing_ok=True)
        source_image = download_source_image(metadata)
        keyframe_image = generate_motion_keyframe(metadata, source_image)
        create_looping_gif(source_image, keyframe_image, cache_file)

    site_file = site_dir / metadata["site_path"]
    site_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_file, site_file)
    _write_site_index(site_dir, metadata)

    write_github_output("available", "true")
    write_github_output("relative_path", metadata["relative_path"])
    write_github_output("cache_hit", str(cache_hit).lower())
    return site_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser("metadata")
    metadata_parser.add_argument("--output", required=True)
    metadata_parser.add_argument("--publication-id")

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--metadata", required=True)
    render_parser.add_argument("--cache-dir", required=True)
    render_parser.add_argument("--site-dir", required=True)
    render_parser.add_argument("--soft-fail", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "metadata":
            prepare_metadata(args.output, publication_id=args.publication_id)
        else:
            render_animation(args.metadata, args.cache_dir, args.site_dir)
    except AnimationError as exc:
        if args.command == "render" and args.soft_fail:
            print(f"WARNING: generated animation unavailable: {exc}", file=sys.stderr)
            write_github_output("available", "false")
            return 0
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
