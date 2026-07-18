import base64
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from PIL import Image, ImageDraw

import generate_top_image as generator


def make_pattern(size=128):
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                (x * 7 + y * 3) % 256,
                (x * 2 + y * 11) % 256,
                (x * 13 + y * 5) % 256,
            )
    return image


def make_motion_pair(size=128):
    source = make_pattern(size)
    keyframe = source.copy()
    box = (size // 4, size // 4, size // 2, size * 3 // 4)
    ImageDraw.Draw(source).rounded_rectangle(box, radius=8, fill=(235, 45, 90))
    shifted = (box[0] + 6, box[1] - 3, box[2] + 6, box[3] - 3)
    ImageDraw.Draw(keyframe).rounded_rectangle(
        shifted, radius=8, fill=(235, 45, 90)
    )
    return source, keyframe

def make_small_motion_pair(size=128):
    source = make_pattern(size)
    keyframe = source.copy()
    left = size // 2 - 8
    top = size // 2 - 8
    patch = source.crop((left, top, left + 16, top + 16))
    keyframe.paste(
        source.crop((left - 3, top - 3, left + 13, top + 13)), (left, top)
    )
    keyframe.paste(patch, (left + 3, top + 2))
    return source, keyframe


def image_bytes(image, image_format="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class GenerateTopImageTests(unittest.TestCase):
    def test_build_metadata_is_deterministic_and_uses_unique_publication_path(self):
        track = {
            "id": "track-1",
            "name": "Song",
            "artist": "Artist",
            "art": "https://i.scdn.co/image/cover-1",
        }
        first = generator.build_metadata(
            track, model="gpt-image-2", quality="medium",
            publication_id="run-100-1",
        )
        second = generator.build_metadata(
            track, model="gpt-image-2", quality="medium",
            publication_id="run-100-1",
        )
        another_run = generator.build_metadata(
            track, model="gpt-image-2", quality="medium",
            publication_id="run-101-1",
        )

        self.assertEqual(first["asset_key"], second["asset_key"])
        self.assertEqual(first["asset_key"], another_run["asset_key"])
        self.assertEqual(len(first["asset_key"]), 24)
        self.assertEqual(first["source_image_url"], track["art"])
        self.assertEqual(
            first["site_path"],
            f"animations/top-track-{first['asset_key']}-run-100-1.gif",
        )
        self.assertEqual(first["relative_path"], first["site_path"])
        self.assertNotEqual(first["site_path"], another_run["site_path"])
        self.assertEqual(first["render_version"], generator.RENDER_VERSION)

        variants = [
            ({**track, "id": "track-2"}, "gpt-image-2", "medium"),
            ({**track, "name": "Another song"}, "gpt-image-2", "medium"),
            ({**track, "art": "https://i.scdn.co/image/cover-2"}, "gpt-image-2", "medium"),
            (track, "another-model", "medium"),
            (track, "gpt-image-2", "low"),
        ]
        for changed_track, model, quality in variants:
            with self.subTest(track=changed_track, model=model, quality=quality):
                changed = generator.build_metadata(
                    changed_track, model=model, quality=quality,
                    publication_id="run-100-1",
                )
                self.assertNotEqual(first["asset_key"], changed["asset_key"])

        with self.assertRaisesRegex(generator.AnimationError, "Publication ID"):
            generator.build_metadata(track, publication_id="../unsafe")

    def test_build_prompt_requests_character_motion_and_locked_cover(self):
        prompt = generator.build_prompt({"name": "Ignored", "artist": "Ignored"})

        self.assertIn("supplied square album cover", prompt)
        self.assertIn("Live2D-style character loop", prompt)
        self.assertIn("Articulate the main", prompt)
        self.assertIn("one rigid layer", prompt)
        self.assertIn("natural blink", prompt)
        self.assertIn("Keep all text unchanged", prompt)
        self.assertIn("camera must remain completely locked", prompt)
        self.assertIn("no pan, zoom", prompt)

    def test_download_source_image_validates_and_decodes_cover(self):
        cover = Image.new("RGB", (48, 48), (20, 40, 80))
        response = Mock(
            status_code=200,
            headers={"Content-Type": "image/jpeg"},
            content=image_bytes(cover, "JPEG"),
        )

        with patch.object(generator.requests, "get", return_value=response) as get:
            image = generator.download_source_image(
                {"source_image_url": "https://i.scdn.co/image/example"}
            )

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (48, 48))
        get.assert_called_once_with("https://i.scdn.co/image/example", timeout=30)

    def test_download_source_image_rejects_non_square_cover(self):
        cover = Image.new("RGB", (64, 48), (20, 40, 80))
        response = Mock(
            status_code=200,
            headers={"Content-Type": "image/png"},
            content=image_bytes(cover),
        )

        with patch.object(generator.requests, "get", return_value=response):
            with self.assertRaisesRegex(generator.AnimationError, "square"):
                generator.download_source_image(
                    {"source_image_url": "https://i.scdn.co/image/non-square"}
                )

    def test_download_source_image_rejects_non_image_response(self):
        response = Mock(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=b"<html></html>",
        )
        with patch.object(generator.requests, "get", return_value=response):
            with self.assertRaisesRegex(generator.AnimationError, "unexpected type"):
                generator.download_source_image(
                    {"source_image_url": "https://example.test/cover"}
                )

    def test_generate_motion_keyframe_posts_reference_image_and_decodes_png(self):
        source = Image.new("RGB", (32, 32), (20, 40, 80))
        edited = Image.new("RGBA", (64, 64), (80, 40, 20, 255))
        response = Mock(status_code=200)
        response.json.return_value = {
            "data": [{
                "b64_json": base64.b64encode(
                    image_bytes(edited)
                ).decode("ascii")
            }]
        }
        metadata = {
            "name": "Song", "artist": "Artist",
            "model": "gpt-image-2", "quality": "medium",
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
            patch.object(generator.requests, "post", return_value=response) as post,
        ):
            image = generator.generate_motion_keyframe(metadata, source)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (64, 64))
        args, kwargs = post.call_args
        self.assertEqual(args[0], generator.OPENAI_IMAGE_EDIT_URL)
        self.assertEqual(kwargs["timeout"], 300)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["data"]["model"], "gpt-image-2")
        self.assertEqual(kwargs["data"]["size"], "1024x1024")
        self.assertEqual(kwargs["data"]["quality"], "medium")
        self.assertEqual(kwargs["data"]["output_format"], "png")
        self.assertNotIn("input_fidelity", kwargs["data"])
        filename, content, mime_type = kwargs["files"]["image[]"]
        self.assertEqual(filename, "album-cover.png")
        self.assertEqual(mime_type, "image/png")
        self.assertTrue(content.startswith(b"\x89PNG"))

    def test_generate_motion_keyframe_rejects_missing_key_without_request(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            patch.object(generator.requests, "post") as post,
        ):
            with self.assertRaisesRegex(generator.AnimationError, "OPENAI_API_KEY"):
                generator.generate_motion_keyframe(
                    {"name": "Song", "artist": "Artist"}, make_pattern()
                )
        post.assert_not_called()

    def test_small_local_character_motion_is_not_rejected(self):
        source, keyframe = make_small_motion_pair()
        source_array = generator._fit_rgb_array(source, 128)
        keyframe_array = generator._fit_rgb_array(keyframe, 128)
        aligned = generator._align_keyframe(source_array, keyframe_array)

        flow = generator._character_motion_flow(source_array, aligned)
        magnitude = np.linalg.norm(flow, axis=2)

        self.assertGreater(float(magnitude.max()), 0.25)
        self.assertLess(float(np.mean(magnitude > 0.25)), 0.05)

    def test_safe_canvas_centers_complete_cover_at_eighty_five_percent(self):
        expected_sizes = {384: 326, 320: 272, 256: 218, 128: 108}
        for canvas_size, content_size in expected_sizes.items():
            with self.subTest(canvas_size=canvas_size):
                self.assertEqual(
                    generator._safe_content_size(canvas_size), content_size
                )

        source = make_pattern(128)
        content_size = generator._safe_content_size(128)
        content = generator._fit_rgb_array(source, content_size)
        background = generator._safe_canvas_background(source, 128)
        canvas = generator._composite_safe_canvas(background, content)
        margin = (128 - content_size) // 2

        self.assertEqual(canvas.shape, (128, 128, 3))
        np.testing.assert_array_equal(
            canvas[margin : margin + content_size, margin : margin + content_size],
            content,
        )
        np.testing.assert_array_equal(canvas[:margin], background[:margin])
        self.assertLess(
            float(background.mean()),
            float(generator._fit_rgb_array(source, 128).mean()),
        )

    def test_animation_moves_toward_keyframe_and_has_a_smooth_loop_seam(self):
        source, keyframe = make_motion_pair()
        frames = generator._animation_frames(source, keyframe, 128, 8, 128)
        arrays = [
            np.asarray(frame.convert("RGB"), dtype=np.float32) for frame in frames
        ]
        content_size = generator._safe_content_size(128)
        safe_margin = (128 - content_size) // 2
        background_mask = np.ones((128, 128), dtype=bool)
        background_mask[
            safe_margin : safe_margin + content_size,
            safe_margin : safe_margin + content_size,
        ] = False
        np.testing.assert_array_equal(
            arrays[0][background_mask],
            arrays[len(arrays) // 2][background_mask],
        )

        def red_centroid(frame):
            mask = (
                (frame[:, :, 0] > 170)
                & (frame[:, :, 1] < 120)
                & (frame[:, :, 2] < 160)
            )
            y, x = np.nonzero(mask)
            return float(x.mean()), float(y.mean())

        start_x, start_y = red_centroid(arrays[0])
        peak_x, peak_y = red_centroid(arrays[len(arrays) // 2])
        self.assertGreater(peak_x, start_x + 0.25)
        self.assertLess(peak_y, start_y - 0.20)

        first_step = float(np.mean(np.abs(arrays[1] - arrays[0])))
        loop_seam = float(np.mean(np.abs(arrays[-1] - arrays[0])))
        self.assertAlmostEqual(loop_seam, first_step, places=5)

    def test_motion_flow_rejects_a_globally_redrawn_cover(self):
        source = np.asarray(make_pattern(), dtype=np.uint8)
        keyframe = np.full_like(source, 255)

        with self.assertRaisesRegex(generator.AnimationError, "redrew too much"):
            generator._character_motion_flow(source, keyframe)

    def test_alignment_failure_does_not_use_an_unaligned_keyframe(self):
        source = np.asarray(make_pattern(), dtype=np.uint8)
        with patch.object(
            generator.cv2, "findTransformECC",
            side_effect=generator.cv2.error("alignment failed"),
        ):
            with self.assertRaisesRegex(generator.AnimationError, "could not be aligned"):
                generator._align_keyframe(source, source.copy())

    def test_alignment_rejects_whole_cover_translation(self):
        source = np.asarray(make_pattern(), dtype=np.uint8)
        translated = np.array(
            [[1.0, 0.0, 8.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        with patch.object(
            generator.cv2, "findTransformECC", return_value=(0.99, translated)
        ):
            with self.assertRaisesRegex(generator.AnimationError, "framing"):
                generator._align_keyframe(source, source.copy())

    def test_create_looping_gif_creates_character_motion_and_removes_oversize(self):
        source, keyframe = make_motion_pair()
        profile = ((96, 8, 32, 50),)
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.gif"
            generator.create_looping_gif(
                source, keyframe, valid_path, profiles=profile
            )

            self.assertTrue(generator.is_valid_animated_gif(valid_path))
            with Image.open(valid_path) as animation:
                self.assertTrue(animation.is_animated)
                self.assertGreater(animation.n_frames, 1)
                self.assertEqual(animation.size, (96, 96))
                self.assertEqual(animation.info.get("loop"), 0)
                decoded = []
                for frame_index in range(animation.n_frames):
                    animation.seek(frame_index)
                    decoded.append(np.asarray(animation.convert("RGB")).copy())
                content_size = generator._safe_content_size(96)
                safe_margin = (96 - content_size) // 2
                background_mask = np.ones((96, 96), dtype=bool)
                background_mask[
                    safe_margin : safe_margin + content_size,
                    safe_margin : safe_margin + content_size,
                ] = False
                np.testing.assert_array_equal(
                    decoded[0][background_mask],
                    decoded[len(decoded) // 2][background_mask],
                )
                frame_delta = np.mean(
                    np.abs(
                        decoded[0].astype(np.int16)
                        - decoded[len(decoded) // 2].astype(np.int16)
                    )
                )
                self.assertGreater(float(frame_delta), 0.0)

            oversized_path = Path(temp_dir) / "oversized.gif"
            with self.assertRaises(generator.AnimationError):
                generator.create_looping_gif(
                    source, keyframe, oversized_path,
                    profiles=profile, max_bytes=1,
                )
            self.assertFalse(oversized_path.exists())
            self.assertFalse((Path(temp_dir) / ".oversized.gif.tmp").exists())

    def test_create_looping_gif_falls_back_to_a_smaller_profile(self):
        source, keyframe = make_motion_pair()
        large_profile = ((128, 8, 64, 50),)
        small_profile = ((64, 8, 32, 50),)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            large_path = root / "large.gif"
            small_path = root / "small.gif"
            generator.create_looping_gif(
                source, keyframe, large_path, profiles=large_profile
            )
            generator.create_looping_gif(
                source, keyframe, small_path, profiles=small_profile
            )
            self.assertGreater(large_path.stat().st_size, small_path.stat().st_size)
            max_bytes = (large_path.stat().st_size + small_path.stat().st_size) // 2

            output_path = root / "fallback.gif"
            generator.create_looping_gif(
                source, keyframe, output_path,
                profiles=large_profile + small_profile,
                max_bytes=max_bytes,
            )

            self.assertLessEqual(output_path.stat().st_size, max_bytes)
            with Image.open(output_path) as animation:
                self.assertEqual(animation.size, (64, 64))

    def test_render_animation_reuses_cache_and_builds_unique_site(self):
        metadata = generator.build_metadata(
            {
                "id": "track-1", "name": "<Track & One>", "artist": "A > B",
                "art": "https://i.scdn.co/image/cover",
            },
            model="gpt-image-2", quality="medium",
            publication_id="run-100-1",
        )
        source, keyframe = make_motion_pair()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )
            cache_dir = root / "cache"
            cache_dir.mkdir()
            cache_file = cache_dir / metadata["filename"]
            generator.create_looping_gif(
                source, keyframe, cache_file,
                profiles=((96, 8, 32, 50),),
            )
            site_dir = root / "site"
            output_file = root / "github-output.txt"

            with (
                patch.dict(
                    os.environ, {"GITHUB_OUTPUT": str(output_file)}, clear=False
                ),
                patch.object(generator, "download_source_image") as download,
                patch.object(generator, "generate_motion_keyframe") as generate,
            ):
                result = generator.render_animation(
                    metadata_path, cache_dir, site_dir
                )

            download.assert_not_called()
            generate.assert_not_called()
            self.assertEqual(result, site_dir / metadata["site_path"])
            self.assertTrue(generator.is_valid_animated_gif(result))
            self.assertTrue((site_dir / ".nojekyll").exists())
            index = (site_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("&lt;Track &amp; One&gt;", index)
            self.assertIn(metadata["relative_path"], index)
            outputs = output_file.read_text(encoding="utf-8")
            self.assertIn("available=true", outputs)
            self.assertIn(f"relative_path={metadata['relative_path']}", outputs)
            self.assertIn("cache_hit=true", outputs)

    def test_render_animation_builds_cache_from_cover_and_keyframe(self):
        metadata = generator.build_metadata(
            {
                "id": "track-1", "name": "Track", "artist": "Artist",
                "art": "https://i.scdn.co/image/cover",
            },
            model="gpt-image-2", quality="medium",
            publication_id="run-100-1",
        )
        source, keyframe = make_motion_pair()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with (
                patch.object(
                    generator, "download_source_image",
                    return_value=source,
                ) as download,
                patch.object(
                    generator, "generate_motion_keyframe",
                    return_value=keyframe,
                ) as generate,
                patch.object(generator, "GIF_PROFILES", ((96, 8, 32, 50),)),
            ):
                result = generator.render_animation(
                    metadata_path, root / "cache", root / "site"
                )

            download.assert_called_once_with(metadata)
            generate.assert_called_once_with(metadata, source)
            self.assertTrue(generator.is_valid_animated_gif(result))

    def test_soft_fail_returns_success_and_marks_animation_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "github-output.txt"
            with (
                patch.dict(
                    os.environ, {"GITHUB_OUTPUT": str(output_file)}, clear=False
                ),
                patch.object(
                    generator, "render_animation",
                    side_effect=generator.AnimationError("expected"),
                ),
            ):
                result = generator.main([
                    "render", "--metadata", "metadata.json",
                    "--cache-dir", "cache", "--site-dir", "site",
                    "--soft-fail",
                ])

            self.assertEqual(result, 0)
            self.assertIn(
                "available=false", output_file.read_text(encoding="utf-8")
            )


    def test_soft_fail_does_not_hide_unexpected_programming_errors(self):
        with patch.object(
            generator, "render_animation", side_effect=TypeError("unexpected")
        ):
            with self.assertRaises(TypeError):
                generator.main([
                    "render", "--metadata", "metadata.json",
                    "--cache-dir", "cache", "--site-dir", "site",
                    "--soft-fail",
                ])

if __name__ == "__main__":
    unittest.main()
