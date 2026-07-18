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


if __name__ == "__main__":
    unittest.main()
