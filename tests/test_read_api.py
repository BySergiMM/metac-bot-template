"""The read-only API client: URL building, retries, and the write ban."""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
import zipfile

from research import metaculus_read_api as api


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _client(**kwargs) -> api.MetaculusReadClient:
    kwargs.setdefault("sleep", lambda _seconds: None)
    kwargs.setdefault("sleep_between_requests", 0)
    return api.MetaculusReadClient("token-abc", **kwargs)


class ConstructionTests(unittest.TestCase):
    def test_token_is_required(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                api.MetaculusReadClient(bad)  # type: ignore[arg-type]

    def test_base_url_trailing_slash_is_normalised(self):
        client = _client(base_url="https://example.com/api/")
        self.assertEqual(client.base_url, "https://example.com/api")


class UrlTests(unittest.TestCase):
    def test_repeated_list_params(self):
        """``post_ids`` must be repeated query params -- the server reads them
        with ``request.GET.getlist('post_ids')``."""
        url = _client()._build_url("/data/download/", {"post_ids": [1, 2, 3]})
        self.assertIn("post_ids=1&post_ids=2&post_ids=3", url)

    def test_booleans_are_lowercased(self):
        url = _client()._build_url("/data/download/", {"include_scores": True, "include_comments": False})
        self.assertIn("include_scores=true", url)
        self.assertIn("include_comments=false", url)

    def test_none_values_are_dropped(self):
        url = _client()._build_url("/posts/", {"a": None, "b": 1})
        self.assertNotIn("a=", url)
        self.assertIn("b=1", url)

    def test_no_params_leaves_a_bare_url(self):
        self.assertEqual(_client()._build_url("/users/me/", None), api.DEFAULT_BASE_URL + "/users/me/")


class ReadOnlyTests(unittest.TestCase):
    def test_non_get_methods_are_refused(self):
        """The lab must be structurally incapable of writing to Metaculus."""
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with self.assertRaises(RuntimeError) as ctx:
                _client()._request("/questions/forecast/", None, method=method)
            self.assertIn("read-only", str(ctx.exception))

    def test_client_exposes_no_write_helpers(self):
        """Prefix match, not substring: ``get_posts_forecasted_by`` is a read.
        What must not exist is a method whose *name begins* with a write verb,
        which is how forecasting-tools names its write side
        (``post_binary_question_prediction``, ``post_question_comment``,
        ``resolve_question``)."""
        names = [name for name in dir(api.MetaculusReadClient) if not name.startswith("_")]
        write_verbs = ("post_", "create_", "submit_", "resolve_", "delete_", "update_", "publish_", "withdraw_")
        offenders = [name for name in names if name.lower().startswith(write_verbs)]
        self.assertEqual(offenders, [], "unexpected write-shaped method(s): {0}".format(offenders))


class RequestBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.calls: list[str] = []
        self.original = api.urllib.request.urlopen
        self.addCleanup(setattr, api.urllib.request, "urlopen", self.original)

    def _install(self, responses):
        queue = list(responses)

        def fake_urlopen(request, timeout=None):  # noqa: ARG001
            self.calls.append(request.full_url)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        api.urllib.request.urlopen = fake_urlopen

    def test_authorization_header_is_sent(self):
        captured = {}

        def fake_urlopen(request, timeout=None):  # noqa: ARG001
            captured.update(request.headers)
            return _FakeResponse(b"{}")

        api.urllib.request.urlopen = fake_urlopen
        _client().get_json("/users/me/")
        # urllib title-cases header names
        self.assertEqual(captured.get("Authorization"), "Token token-abc")
        self.assertIn("research", captured.get("User-agent", ""))

    def test_retries_on_429_then_succeeds(self):
        self._install(
            [
                urllib.error.HTTPError("u", 429, "rate limited", {}, io.BytesIO(b"slow down")),
                _FakeResponse(json.dumps({"id": 7}).encode()),
            ]
        )
        self.assertEqual(_client().get_json("/users/me/"), {"id": 7})
        self.assertEqual(len(self.calls), 2)

    def test_does_not_retry_a_403(self):
        """A permission error is not transient; retrying only burns the rate
        limit and hides the real message."""
        self._install(
            [urllib.error.HTTPError("u", 403, "forbidden", {}, io.BytesIO(b"no access for you"))]
        )
        with self.assertRaises(api.MetaculusReadError) as ctx:
            _client().get_json("/data/download/")
        self.assertEqual(ctx.exception.status, 403)
        self.assertIn("no access for you", ctx.exception.body or "")
        self.assertEqual(len(self.calls), 1)

    def test_gives_up_after_max_retries(self):
        self._install([urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"")) for _ in range(4)])
        with self.assertRaises(api.MetaculusReadError):
            _client(max_retries=4).get_json("/posts/")
        self.assertEqual(len(self.calls), 4)

    def test_call_log_records_every_request(self):
        self._install([_FakeResponse(b"{}")])
        client = _client()
        client.get_json("/users/me/")
        self.assertEqual(client.call_log[-1]["path"], "/users/me/")
        self.assertEqual(client.call_log[-1]["status"], 200)


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.original = api.urllib.request.urlopen
        self.addCleanup(setattr, api.urllib.request, "urlopen", self.original)

    def _install_pages(self, pages):
        queue = list(pages)

        def fake_urlopen(request, timeout=None):  # noqa: ARG001
            return _FakeResponse(json.dumps(queue.pop(0)).encode())

        api.urllib.request.urlopen = fake_urlopen

    def test_follows_next_until_exhausted(self):
        self._install_pages(
            [
                {"results": [{"id": 1}, {"id": 2}], "next": "more"},
                {"results": [{"id": 3}], "next": None},
            ]
        )
        posts = list(_client().iter_posts({"tournaments": "x"}))
        self.assertEqual([p["id"] for p in posts], [1, 2, 3])

    def test_deduplicates_overlapping_pages(self):
        """Overlapping pages would otherwise double-count questions in the
        coverage denominator."""
        self._install_pages(
            [
                {"results": [{"id": 1}, {"id": 2}], "next": "more"},
                {"results": [{"id": 2}, {"id": 3}], "next": None},
            ]
        )
        posts = list(_client().iter_posts({"tournaments": "x"}))
        self.assertEqual([p["id"] for p in posts], [1, 2, 3])

    def test_stops_at_max_pages(self):
        def fake_urlopen(request, timeout=None):  # noqa: ARG001
            return _FakeResponse(json.dumps({"results": [{"id": 1}], "next": "always"}).encode())

        api.urllib.request.urlopen = fake_urlopen
        posts = list(_client().iter_posts({"tournaments": "x"}, max_pages=3))
        # deduplicated to one, but crucially it terminated
        self.assertEqual(len(posts), 1)

    def test_rejects_a_non_object_payload(self):
        self._install_pages([["not", "a", "dict"]])
        with self.assertRaises(api.MetaculusReadError):
            list(_client().iter_posts({"tournaments": "x"}))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.original = api.urllib.request.urlopen
        self.addCleanup(setattr, api.urllib.request, "urlopen", self.original)

    def test_returns_zip_bytes(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("question_data.csv", "Question ID\n1\n")
        payload = buffer.getvalue()

        api.urllib.request.urlopen = lambda request, timeout=None: _FakeResponse(
            payload, headers={"Content-Type": "application/zip"}
        )
        self.assertEqual(_client().download_data_zip([1, 2]), payload)

    def test_rejects_a_non_zip_response(self):
        """A JSON error body with a 200 would otherwise be written to disk as
        if it were data."""
        api.urllib.request.urlopen = lambda request, timeout=None: _FakeResponse(
            b'{"detail": "No questions found"}'
        )
        with self.assertRaises(api.MetaculusReadError) as ctx:
            _client().download_data_zip([1])
        self.assertIn("expected a zip", str(ctx.exception))

    def test_empty_post_ids_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            _client().download_data_zip([])

    def test_user_ids_is_never_sent(self):
        """The serializer rejects ``user_ids`` for accounts without the data
        tier, and the server already scopes the export to us."""
        captured = {}

        def fake_urlopen(request, timeout=None):  # noqa: ARG001
            captured["url"] = request.full_url
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("question_data.csv", "Question ID\n")
            return _FakeResponse(buffer.getvalue(), headers={"Content-Type": "application/zip"})

        api.urllib.request.urlopen = fake_urlopen
        _client().download_data_zip([1])
        self.assertNotIn("user_ids", captured["url"])


class HelperTests(unittest.TestCase):
    def test_chunked(self):
        self.assertEqual(list(api.chunked([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(list(api.chunked([], 2)), [])

    def test_long_id_lists_are_summarised_in_the_log(self):
        summarised = api._loggable({"post_ids": list(range(50))})
        self.assertIn("50 values", summarised["post_ids"])

    def test_short_params_are_logged_verbatim(self):
        self.assertEqual(api._loggable({"post_ids": [1, 2]}), {"post_ids": [1, 2]})


if __name__ == "__main__":
    unittest.main()
