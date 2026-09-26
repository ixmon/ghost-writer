"""Chapter list, scores, and version restore."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class ChapterApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.library = Path(self._tmp.name)
        self.previous = server.LIBRARY_DIR
        server.LIBRARY_DIR = str(self.library)
        book = self.library / "demo"
        book.mkdir()
        (book / "demo.yaml").write_text(
            "\n".join([
                "title: Demo",
                "chapters:",
                "  - number: 1",
                "    title: Arrival",
                "    synopsis: They arrive.",
                "    pov_character: Ada",
                "    emotional_beat: dread",
                "    location: the dock",
                "    key_events:",
                "      - knock on the hull",
                "  - number: 2",
                "    title: The lock",
                "    synopsis: She decides.",
                "    key_events:",
                "      - turn the key",
                "",
            ])
        )
        (book / "chapter_01.md").write_text("Ada waited in the rain.\n")
        (book / "chapter_01_v1.md").write_text("Older draft.\n")
        (book / "review_01.md").write_text("The opening is thin.\n\nSCORE: 72/100\n")
        self.client = TestClient(server.app)

    def tearDown(self):
        server.LIBRARY_DIR = self.previous

    def test_library_counts_chapters_that_are_only_in_the_outline(self):
        response = self.client.get("/api/books")
        self.assertEqual(response.status_code, 200)
        demo = next(book for book in response.json()["books"] if book["name"] == "demo")
        self.assertEqual(demo["chapter_count"], 2)

    def test_outline_chapters_include_unwritten_plans_and_scores(self):
        response = self.client.get("/api/books/demo/chapters")
        self.assertEqual(response.status_code, 200)
        chapters = response.json()["chapters"]
        self.assertEqual([chapter["number"] for chapter in chapters], [1, 2])
        self.assertEqual(chapters[0]["status"], "reviewed")
        self.assertEqual(chapters[0]["score"], 72)
        self.assertEqual(chapters[0]["title"], "Arrival")
        self.assertEqual(chapters[0]["versions"], 1)
        self.assertEqual(chapters[1]["status"], "planned")
        self.assertIsNone(chapters[1]["score"])
        self.assertEqual(chapters[1]["synopsis"], "She decides.")
        self.assertIsNone(chapters[1]["file"])

    def test_unwritten_chapter_opens_with_its_plan(self):
        response = self.client.get("/api/books/demo/chapters/2")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["exists"])
        self.assertEqual(body["content"], "")
        self.assertEqual(body["outline"]["title"], "The lock")
        self.assertEqual(body["outline"]["key_events"], ["turn the key"])

    def test_save_keeps_a_version_and_restore_brings_it_back(self):
        saved = self.client.put(
            "/api/books/demo/chapters/1",
            json={"content": "Ada waited, then opened the hatch.\n"},
        )
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["backup"], "chapter_01_v2.md")

        version = self.client.get("/api/books/demo/chapters/1/versions/2")
        self.assertEqual(version.status_code, 200)
        self.assertIn("rain", version.json()["content"])

        restored = self.client.post(
            "/api/books/demo/chapters/1/restore",
            json={"version": 1},
        )
        self.assertEqual(restored.status_code, 200)
        current = self.client.get("/api/books/demo/chapters/1")
        self.assertIn("Older draft", current.json()["content"])

    def test_review_payload_includes_the_score(self):
        response = self.client.get("/api/books/demo/reviews/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 72)


if __name__ == "__main__":
    unittest.main()
