import unittest
from unittest.mock import Mock, patch

import refresh_widget


class RefreshWidgetTests(unittest.TestCase):
    def test_get_top_tracks_keeps_spotify_id(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "items": [
                {
                    "id": "track-123",
                    "name": "Song",
                    "artists": [{"name": "Artist"}],
                    "album": {
                        "images": [
                            {
                                "url": "https://example.test/art.jpg",
                                "width": 640,
                                "height": 640,
                            }
                        ]
                    },
                }
            ]
        }

        with patch.object(refresh_widget.requests, "get", return_value=response):
            tracks = refresh_widget.get_top_tracks(
                "access-token", "short_term", limit=1
            )

        self.assertEqual(tracks[0]["id"], "track-123")
        self.assertEqual(tracks[0]["art"], "https://example.test/art.jpg")

    def test_build_payload_prefers_configured_top_image(self):
        tracks = [
            {
                "id": "track-123",
                "name": "Song",
                "artist": "Artist",
                "art": "https://example.test/album.jpg",
            }
        ]
        payload = refresh_widget.build_payload(
            {
                "top_image_url": "https://example.test/generated.gif",
                "track_count": 1,
            },
            tracks,
        )

        fields = {
            item["name"]: item["value"]
            for item in payload["data"]["dynamic"]
        }
        self.assertEqual(
            fields["top_image"]["url"],
            "https://example.test/generated.gif",
        )

    def test_build_payload_falls_back_to_album_art(self):
        tracks = [
            {
                "id": "track-123",
                "name": "Song",
                "artist": "Artist",
                "art": "https://example.test/album.jpg",
            }
        ]
        payload = refresh_widget.build_payload(
            {"top_image_url": "", "track_count": 1}, tracks
        )

        fields = {
            item["name"]: item["value"]
            for item in payload["data"]["dynamic"]
        }
        self.assertEqual(
            fields["top_image"]["url"],
            "https://example.test/album.jpg",
        )


    def test_top_image_unfurl_status_requires_complete_media_metadata(self):
        url = "https://example.test/generated.gif"
        response_data = {
            "data": {
                "dynamic": [
                    {
                        "type": 3,
                        "name": "top_image",
                        "value": {
                            "url": url,
                            "proxy_url": "https://proxy.example/image.gif",
                            "width": 384,
                            "height": 384,
                            "content_type": "image/gif",
                            "loading_state": 2,
                        },
                    }
                ]
            }
        }

        status = refresh_widget.top_image_unfurl_status(
            response_data, expected_url=url
        )

        self.assertTrue(status["present"])
        self.assertTrue(status["ready"])
        self.assertTrue(status["source_matches"])
        self.assertTrue(status["has_proxy"])

    def test_top_image_unfurl_status_rejects_failed_placeholder(self):
        response_data = {
            "data": {
                "dynamic": [
                    {
                        "type": 3,
                        "name": "top_image",
                        "value": {
                            "url": "https://example.test/generated.gif",
                            "proxy_url": "https://proxy.example/image.gif",
                            "width": None,
                            "height": None,
                            "content_type": None,
                            "loading_state": 3,
                        },
                    }
                ]
            }
        }

        status = refresh_widget.top_image_unfurl_status(response_data)

        self.assertTrue(status["present"])
        self.assertFalse(status["ready"])
        self.assertEqual(status["loading_state"], 3)
        self.assertTrue(status["has_proxy"])

    def test_log_top_image_unfurl_handles_empty_patch_response(self):
        response = Mock()
        response.json.side_effect = ValueError("no json")

        with patch("builtins.print") as print_mock:
            status = refresh_widget.log_top_image_unfurl(response)

        self.assertIsNone(status)
        self.assertIn("without media metadata", print_mock.call_args.args[0])

if __name__ == "__main__":
    unittest.main()
