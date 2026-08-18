"""Dataset identity, hashing and provenance manifests.

A dataset is a directory whose name *is* its identifier, containing the raw
files exactly as the server sent them plus a ``manifest.json`` describing where
they came from. Nothing in the lab is allowed to report a number that cannot be
traced back to one of these directories.

Two properties matter and are enforced here:

- **Immutability.** ``write_manifest`` refuses to overwrite an existing
  dataset. Re-running a fetch produces a new directory; it never silently
  mutates an old one. ``verify_dataset`` re-hashes the files so tampering (or
  bit-rot, or a half-written download) is detectable after the fact.
- **Content addressing.** The dataset id embeds a digest of the file contents,
  so two fetches that returned byte-identical data are visibly the same data
  even though they happened at different times.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

_CHUNK = 1024 * 1024


def utc_now_iso() -> str:
    """Timestamp used for ``created_at``. Second precision, always UTC, always
    with an explicit ``Z`` so it can never be misread as local time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_stamp(iso: str) -> str:
    return re.sub(r"[-:]", "", iso).replace("Z", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_digest(files: list["FileRecord"]) -> str:
    """A single digest over a set of files.

    Sorted by name so the result depends on the *contents* of the dataset and
    not on the order the downloader happened to write things in. Two fetches of
    unchanged data therefore collide on purpose.
    """
    parts = ["{0}:{1}".format(f.name, f.sha256) for f in sorted(files, key=lambda f: f.name)]
    return sha256_bytes("\n".join(parts).encode("utf-8"))


def make_dataset_id(kind: str, created_at: str, digest: str) -> str:
    """e.g. ``track-record-20260819T081500Z-1a2b3c4d``.

    Timestamp first so directory listings sort chronologically; content digest
    last so an unchanged re-fetch is recognisable at a glance.
    """
    return "{0}-{1}-{2}".format(kind, _compact_stamp(created_at), digest[:8])


@dataclass
class FileRecord:
    name: str
    sha256: str
    bytes: int
    rows: int | None = None  # data rows excluding the header, for CSVs

    @classmethod
    def from_path(cls, path: str, rows: int | None = None) -> "FileRecord":
        return cls(
            name=os.path.basename(path),
            sha256=sha256_file(path),
            bytes=os.path.getsize(path),
            rows=rows,
        )


def git_info(repo_root: str) -> dict[str, Any]:
    """Best-effort git state of the tree that produced the dataset.

    Never raises: a dataset fetched from a tarball with no ``.git`` is still a
    valid dataset, it just has less provenance. ``dirty`` matters more than
    ``commit`` -- a dirty tree means the commit alone does not identify the code
    that ran.
    """

    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:
            return None
        if out.returncode != 0:
            return None
        return out.stdout.decode("utf-8", "replace").strip()

    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "available": commit is not None,
    }


@dataclass
class Manifest:
    """Everything needed to answer "where did this number come from?"."""

    dataset_id: str
    kind: str
    created_at: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    tool: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    account: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    files: list[FileRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["files"] = [asdict(f) for f in self.files]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        data = dict(data)
        data["files"] = [FileRecord(**f) for f in data.get("files", [])]
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError("unknown manifest fields: {0}".format(sorted(unknown)))
        return cls(**data)


def write_manifest(dataset_dir: str, manifest: Manifest) -> str:
    """Write ``manifest.json``. Refuses to clobber an existing one.

    Overwriting a manifest would silently invalidate every result already
    attributed to that dataset id, which is exactly the failure mode datasets
    exist to prevent.
    """
    path = os.path.join(dataset_dir, MANIFEST_NAME)
    if os.path.exists(path):
        raise FileExistsError(
            "refusing to overwrite existing manifest at {0}; "
            "datasets are immutable, fetch a new one instead".format(path)
        )
    os.makedirs(dataset_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_manifest(dataset_dir: str) -> Manifest:
    path = os.path.join(dataset_dir, MANIFEST_NAME)
    with open(path) as handle:
        return Manifest.from_dict(json.load(handle))


def verify_dataset(dataset_dir: str) -> list[str]:
    """Re-hash every file named in the manifest. Returns a list of problems;
    an empty list means the dataset on disk is exactly what was downloaded."""
    problems: list[str] = []
    try:
        manifest = read_manifest(dataset_dir)
    except FileNotFoundError:
        return ["no manifest.json in {0}".format(dataset_dir)]
    except Exception as exc:  # noqa: BLE001 - any unreadable manifest is a problem
        return ["unreadable manifest: {0}".format(exc)]

    for record in manifest.files:
        path = os.path.join(dataset_dir, record.name)
        if not os.path.exists(path):
            problems.append("missing file: {0}".format(record.name))
            continue
        actual_size = os.path.getsize(path)
        if actual_size != record.bytes:
            problems.append(
                "size mismatch for {0}: manifest {1}, on disk {2}".format(
                    record.name, record.bytes, actual_size
                )
            )
        actual_hash = sha256_file(path)
        if actual_hash != record.sha256:
            problems.append(
                "sha256 mismatch for {0}: manifest {1}, on disk {2}".format(
                    record.name, record.sha256, actual_hash
                )
            )

    expected = content_digest(manifest.files)
    if not manifest.dataset_id.endswith(expected[:8]):
        problems.append(
            "dataset_id {0} does not match content digest {1}".format(
                manifest.dataset_id, expected[:8]
            )
        )
    return problems


def iter_datasets(root: str, kind: str | None = None) -> list[tuple[str, Manifest]]:
    """``(directory, manifest)`` pairs under ``root``, newest first.

    The directory path is carried explicitly rather than rebuilt from
    ``dataset_id``. The fetch script does name directories after the id, but a
    dataset that was moved or copied under another name is still a perfectly
    valid dataset, and reconstructing the path would fail on it with a
    confusing FileNotFoundError instead of just working.

    Unreadable directories are skipped rather than raising, so one bad dataset
    cannot hide the rest.
    """
    out: list[tuple[str, Manifest]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            manifest = read_manifest(path)
        except Exception:  # noqa: BLE001 - a broken dataset should not hide good ones
            continue
        if kind and manifest.kind != kind:
            continue
        out.append((path, manifest))
    out.sort(key=lambda pair: pair[1].created_at, reverse=True)
    return out


def list_datasets(root: str, kind: str | None = None) -> list[Manifest]:
    """All dataset manifests under ``root``, newest first."""
    return [manifest for _path, manifest in iter_datasets(root, kind=kind)]


def latest_dataset_dir(root: str, kind: str | None = None) -> str | None:
    datasets = iter_datasets(root, kind=kind)
    if not datasets:
        return None
    return datasets[0][0]
