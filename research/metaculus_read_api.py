"""A deliberately small, strictly read-only Metaculus API client.

Why not reuse ``forecasting_tools.MetaculusClient``?

- It has no support for ``/api/data/download/`` (verified against
  forecasting-tools 0.2.90: the only user endpoints it knows are
  ``users/me``, ``users/me/bots/`` and ``users/me/bots/<id>/token/``), so the
  one endpoint this milestone is built on would have to be written by hand
  anyway.
- It carries the whole production dependency tree (litellm, pydantic,
  streamlit, ...). The lab is supposed to be runnable without installing the
  bot.
- It exposes write methods (``post_binary_question_prediction``,
  ``post_question_comment``, ``resolve_question``). This module physically
  cannot write: it only ever issues GET, and ``_request`` hard-refuses any
  other verb.

Endpoint contracts below were read off the Metaculus server source
(github.com/Metaculus/metaculus, ``utils/urls.py``, ``utils/views.py``,
``utils/serializers.py``, ``utils/csv_utils.py``, ``posts/serializers.py``,
``users/urls.py``) rather than guessed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

DEFAULT_BASE_URL = "https://www.metaculus.com/api"

# Metaculus sits behind a WAF that 403s requests without a plausible
# User-Agent. The existing backtest script learned this the hard way
# (commit 8745bee, "send a real User-Agent so the Metaculus API stops
# returning 403"), so we keep an explicit, honest, identifying UA.
USER_AGENT = "seergiii-bot-research/1.0 (offline track-record analysis; read-only)"

POSTS_PAGE_LIMIT = 100

# Requesting many posts at once is done via repeated ``post_ids`` query
# params, which the server reads with ``request.GET.getlist("post_ids")``.
# Chunked well below any sane URL length limit.
DOWNLOAD_POST_CHUNK = 50


class MetaculusReadError(RuntimeError):
    """An API call failed in a way the caller needs to see, with the server's
    own error body attached -- silent ``None`` returns are how the previous
    backtest harness spent four commits chasing a phantom parsing bug."""

    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class MetaculusReadClient:
    def __init__(
        self,
        token: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 90,
        max_retries: int = 4,
        sleep_between_requests: float = 0.6,
        sleep: Any = time.sleep,
    ) -> None:
        if not token or not token.strip():
            raise ValueError("a Metaculus API token is required")
        self.token = token.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_between_requests = sleep_between_requests
        self._sleep = sleep
        # Recorded so the manifest can state exactly which calls produced the
        # dataset, without the caller having to remember.
        self.call_log: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- core

    def _build_url(self, path: str, params: dict[str, Any] | None) -> str:
        url = self.base_url + path
        if params:
            pairs: list[tuple[str, str]] = []
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    for item in value:
                        pairs.append((key, str(item)))
                elif isinstance(value, bool):
                    pairs.append((key, "true" if value else "false"))
                else:
                    pairs.append((key, str(value)))
            if pairs:
                url += "?" + urllib.parse.urlencode(pairs)
        return url

    def _request(self, path: str, params: dict[str, Any] | None, method: str = "GET") -> tuple[int, bytes, dict[str, str]]:
        if method != "GET":
            raise RuntimeError(
                "MetaculusReadClient is read-only by construction; "
                "refusing method {0}".format(method)
            )
        url = self._build_url(path, params)
        headers = {
            "Authorization": "Token " + self.token,
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        last_status: int | None = None
        last_body: str | None = None
        for attempt in range(self.max_retries):
            if self.sleep_between_requests:
                self._sleep(self.sleep_between_requests)
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    self.call_log.append(
                        {"path": path, "params": _loggable(params), "status": response.status}
                    )
                    return response.status, body, dict(response.headers)
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                try:
                    last_body = exc.read().decode("utf-8", "replace")[:2000]
                except Exception:  # noqa: BLE001
                    last_body = None
                # 429 and 5xx are transient. 4xx otherwise means we asked the
                # wrong question and retrying just wastes the rate limit.
                if exc.code == 429 or exc.code >= 500:
                    self._sleep(2.0 * (attempt + 1))
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - network layer
                last_status = None
                last_body = str(exc)
                self._sleep(2.0 * (attempt + 1))
                continue

        self.call_log.append(
            {"path": path, "params": _loggable(params), "status": last_status, "error": True}
        )
        raise MetaculusReadError(
            "GET {0} failed (status={1})".format(path, last_status),
            status=last_status,
            body=last_body,
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        _status, body, _headers = self._request(path, params)
        return json.loads(body.decode("utf-8"))

    def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> tuple[bytes, dict[str, str]]:
        _status, body, headers = self._request(path, params)
        return body, headers

    # ------------------------------------------------------------- account

    def get_current_user(self) -> dict[str, Any]:
        """``GET /api/users/me/`` (``users/urls.py``: ``users/me/``).

        Always available to any authenticated account -- Metaculus guarantees
        "for any user account, that account's own data is available to it".
        """
        return self.get_json("/users/me/")

    def get_data_access_status(self) -> Any:
        """``GET /api/get-data-access-status/`` (``misc/urls.py``).

        Tells us which data-access tier this token has. Recorded in the
        manifest because the answer determines which of the reconstructions
        below are possible at all, and it can change without warning.
        """
        return self.get_json("/get-data-access-status/")

    # --------------------------------------------------------------- posts

    def iter_posts(self, params: dict[str, Any], max_pages: int = 200) -> Iterable[dict[str, Any]]:
        """Page through ``GET /api/posts/``.

        Stops on the first empty page or missing ``next``, and hard-stops at
        ``max_pages`` so a server-side paging change can never turn this into
        an infinite loop against the rate limiter.
        """
        offset = 0
        seen_ids: set[int] = set()
        for _page in range(max_pages):
            page_params = dict(params)
            page_params.update({"limit": POSTS_PAGE_LIMIT, "offset": offset})
            data = self.get_json("/posts/", page_params)
            if not isinstance(data, dict):
                raise MetaculusReadError("unexpected /posts/ payload: not an object")
            results = data.get("results") or []
            for post in results:
                post_id = post.get("id")
                # Defensive: overlapping pages would otherwise double-count
                # questions in the coverage report.
                if post_id in seen_ids:
                    continue
                if post_id is not None:
                    seen_ids.add(post_id)
                yield post
            if not results or not data.get("next"):
                return
            offset += POSTS_PAGE_LIMIT

    def get_posts_forecasted_by(self, user_id: int, statuses: list[str] | None = None) -> list[dict[str, Any]]:
        """Every post this account has forecast on.

        ``forecaster_id`` is a real server-side filter
        (``posts/serializers.py``: ``forecaster_id = serializers.IntegerField``,
        wired up in ``posts/views.py``). This is what defines "our track
        record": Metaculus grants resolution access precisely to questions the
        account has forecast on.
        """
        params: dict[str, Any] = {"forecaster_id": user_id, "order_by": "-published_at"}
        if statuses:
            params["statuses"] = statuses
        return list(self.iter_posts(params))

    def get_tournament_posts(self, tournament: str | int, statuses: list[str] | None = None) -> list[dict[str, Any]]:
        """Every post in a tournament, regardless of who forecast on it.

        This is the denominator of the coverage report. Read-only listing of
        question metadata; it does not touch forecasting.
        """
        params: dict[str, Any] = {"tournaments": tournament, "order_by": "-published_at"}
        if statuses:
            params["statuses"] = statuses
        return list(self.iter_posts(params))

    # ------------------------------------------------------------ download

    def download_data_zip(
        self,
        post_ids: list[int],
        include_user_data: bool = True,
        include_scores: bool = True,
        include_comments: bool = False,
        aggregation_methods: str | None = None,
    ) -> bytes:
        """``GET /api/data/download/`` -> a ZIP of CSVs.

        Server-side guarantees, read from ``utils/csv_utils.py``
        (``export_data_for_questions``):

        - ``if not (has_data_access or is_staff):
             user_forecasts = user_forecasts.filter(author=user)`` -- a plain
          account can only ever receive **its own** forecasts, even with
          ``include_user_data=true``. We cannot obtain another forecaster's
          private data through this endpoint, by construction.
        - scores are filtered the same way: our own rows plus rows where
          ``user`` is null (the aggregate's own scores).

        ``user_ids`` is deliberately NOT sent: the serializer rejects it for
        accounts without the data-access tier, and we do not need it -- the
        server already scopes to us.
        """
        if not post_ids:
            raise ValueError("post_ids must not be empty")
        params: dict[str, Any] = {
            "post_ids": list(post_ids),
            "include_user_data": include_user_data,
            "include_scores": include_scores,
            "include_comments": include_comments,
        }
        if aggregation_methods:
            params["aggregation_methods"] = aggregation_methods
        body, headers = self.get_bytes("/data/download/", params)
        content_type = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
        if not body[:2] == b"PK" and "zip" not in content_type:
            raise MetaculusReadError(
                "expected a zip from /data/download/, got content-type={0!r}".format(content_type),
                body=body[:500].decode("utf-8", "replace"),
            )
        return body


def _loggable(params: dict[str, Any] | None) -> dict[str, Any]:
    """Params as recorded in the manifest. Long id lists are summarised so the
    manifest stays readable, and nothing secret is ever in params anyway (the
    token travels in a header)."""
    if not params:
        return {}
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)) and len(value) > 6:
            out[key] = "<{0} values: {1}...>".format(len(value), ",".join(str(v) for v in list(value)[:3]))
        else:
            out[key] = value
    return out


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
