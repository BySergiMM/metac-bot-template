"""Provenance: hashing, dataset identity, immutability, tamper detection."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from research.provenance import (
    FileRecord,
    Manifest,
    content_digest,
    list_datasets,
    latest_dataset_dir,
    make_dataset_id,
    read_manifest,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
    verify_dataset,
    write_manifest,
)


class HashingTests(unittest.TestCase):
    def test_sha256_bytes_matches_known_vector(self):
        # The canonical SHA-256 of the empty string; if this ever changes the
        # problem is far bigger than this repo.
        self.assertEqual(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

    def test_sha256_file_matches_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.txt")
            payload = b"forecast,0.42\n"
            with open(path, "wb") as handle:
                handle.write(payload)
            self.assertEqual(sha256_file(path), sha256_bytes(payload))

    def test_sha256_file_handles_multi_chunk_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.bin")
            payload = b"x" * (3 * 1024 * 1024 + 7)
            with open(path, "wb") as handle:
                handle.write(payload)
            self.assertEqual(sha256_file(path), sha256_bytes(payload))


class DatasetIdTests(unittest.TestCase):
    def test_content_digest_is_order_independent(self):
        a = FileRecord(name="a.csv", sha256="aa", bytes=1)
        b = FileRecord(name="b.csv", sha256="bb", bytes=2)
        self.assertEqual(content_digest([a, b]), content_digest([b, a]))

    def test_content_digest_changes_when_contents_change(self):
        a = FileRecord(name="a.csv", sha256="aa", bytes=1)
        a2 = FileRecord(name="a.csv", sha256="ac", bytes=1)
        self.assertNotEqual(content_digest([a]), content_digest([a2]))

    def test_dataset_id_embeds_timestamp_and_digest(self):
        dataset_id = make_dataset_id("track-record", "2026-08-19T08:15:00Z", "abcdef1234")
        self.assertEqual(dataset_id, "track-record-20260819T081500Z-abcdef12")

    def test_identical_content_yields_identical_id_suffix(self):
        """Reproducibility: the same bytes must be recognisable as the same
        data even when fetched at a different moment."""
        files = [FileRecord(name="a.csv", sha256="aa", bytes=1)]
        first = make_dataset_id("track-record", "2026-08-19T08:15:00Z", content_digest(files))
        second = make_dataset_id("track-record", "2026-09-01T23:59:59Z", content_digest(files))
        self.assertNotEqual(first, second)
        self.assertEqual(first.split("-")[-1], second.split("-")[-1])

    def test_utc_now_iso_is_zulu(self):
        stamp = utc_now_iso()
        self.assertTrue(stamp.endswith("Z"), stamp)
        self.assertEqual(len(stamp), len("2026-08-19T08:15:00Z"))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _dataset(self, name="ds"):
        directory = os.path.join(self.tmp, name)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "question_data.csv")
        with open(path, "w") as handle:
            handle.write("Question ID\n1\n")
        record = FileRecord.from_path(path, rows=1)
        created = utc_now_iso()
        dataset_id = make_dataset_id("track-record", created, content_digest([record]))
        manifest = Manifest(
            dataset_id=dataset_id,
            kind="track-record",
            created_at=created,
            files=[record],
            account={"user_id": 1, "username": "bot"},
        )
        return directory, manifest

    def test_round_trip(self):
        _directory, manifest = self._dataset()
        restored = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
        self.assertEqual(restored.dataset_id, manifest.dataset_id)
        self.assertEqual(restored.files[0].sha256, manifest.files[0].sha256)
        self.assertEqual(restored.account["username"], "bot")

    def test_from_dict_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            Manifest.from_dict(
                {"dataset_id": "x", "kind": "k", "created_at": "t", "surprise": 1}
            )

    def test_write_refuses_to_overwrite(self):
        directory, manifest = self._dataset()
        write_manifest(directory, manifest)
        with self.assertRaises(FileExistsError):
            write_manifest(directory, manifest)

    def test_verify_passes_on_untouched_dataset(self):
        directory, manifest = self._dataset()
        write_manifest(directory, manifest)
        self.assertEqual(verify_dataset(directory), [])

    def test_verify_detects_tampering(self):
        directory, manifest = self._dataset()
        write_manifest(directory, manifest)
        with open(os.path.join(directory, "question_data.csv"), "a") as handle:
            handle.write("2\n")
        problems = verify_dataset(directory)
        self.assertTrue(any("sha256 mismatch" in p for p in problems), problems)

    def test_verify_detects_missing_file(self):
        directory, manifest = self._dataset()
        write_manifest(directory, manifest)
        os.remove(os.path.join(directory, "question_data.csv"))
        problems = verify_dataset(directory)
        self.assertTrue(any("missing file" in p for p in problems), problems)

    def test_verify_detects_id_not_matching_contents(self):
        directory, manifest = self._dataset()
        manifest.dataset_id = "track-record-20260101T000000Z-deadbeef"
        write_manifest(directory, manifest)
        problems = verify_dataset(directory)
        self.assertTrue(any("does not match content digest" in p for p in problems), problems)

    def test_verify_reports_missing_manifest(self):
        directory = os.path.join(self.tmp, "empty")
        os.makedirs(directory)
        problems = verify_dataset(directory)
        self.assertTrue(any("no manifest.json" in p for p in problems), problems)

    def test_listing_orders_newest_first_and_skips_broken(self):
        for name, created in (("old", "2026-01-01T00:00:00Z"), ("new", "2026-08-01T00:00:00Z")):
            directory = os.path.join(self.tmp, name)
            os.makedirs(directory)
            path = os.path.join(directory, "question_data.csv")
            with open(path, "w") as handle:
                handle.write("Question ID\n1\n")
            record = FileRecord.from_path(path)
            write_manifest(
                directory,
                Manifest(
                    dataset_id=make_dataset_id("track-record", created, content_digest([record])),
                    kind="track-record",
                    created_at=created,
                    files=[record],
                ),
            )
        broken = os.path.join(self.tmp, "broken")
        os.makedirs(broken)
        with open(os.path.join(broken, "manifest.json"), "w") as handle:
            handle.write("{not json")

        datasets = list_datasets(self.tmp, kind="track-record")
        self.assertEqual(len(datasets), 2)
        self.assertEqual(datasets[0].created_at, "2026-08-01T00:00:00Z")
        self.assertIsNotNone(latest_dataset_dir(self.tmp, kind="track-record"))

    def test_listing_filters_by_kind(self):
        directory = os.path.join(self.tmp, "other")
        os.makedirs(directory)
        path = os.path.join(directory, "x.csv")
        with open(path, "w") as handle:
            handle.write("a\n")
        record = FileRecord.from_path(path)
        created = utc_now_iso()
        write_manifest(
            directory,
            Manifest(
                dataset_id=make_dataset_id("something-else", created, content_digest([record])),
                kind="something-else",
                created_at=created,
                files=[record],
            ),
        )
        self.assertEqual(list_datasets(self.tmp, kind="track-record"), [])
        self.assertIsNone(latest_dataset_dir(self.tmp, kind="track-record"))

    def test_latest_dataset_dir_works_when_the_folder_was_renamed(self):
        """A dataset copied or moved under a different name is still a valid
        dataset; rebuilding the path from dataset_id would break on it."""
        directory = os.path.join(self.tmp, "renamed-by-hand")
        os.makedirs(directory)
        path = os.path.join(directory, "question_data.csv")
        with open(path, "w") as handle:
            handle.write("Question ID\n1\n")
        record = FileRecord.from_path(path)
        created = utc_now_iso()
        write_manifest(
            directory,
            Manifest(
                dataset_id=make_dataset_id("track-record", created, content_digest([record])),
                kind="track-record",
                created_at=created,
                files=[record],
            ),
        )
        self.assertEqual(latest_dataset_dir(self.tmp, kind="track-record"), directory)

    def test_read_manifest_round_trips_from_disk(self):
        directory, manifest = self._dataset()
        write_manifest(directory, manifest)
        self.assertEqual(read_manifest(directory).dataset_id, manifest.dataset_id)


if __name__ == "__main__":
    unittest.main()
