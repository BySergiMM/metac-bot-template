"""Tests for the offline research lab and the production forecasting path.

Run from the repository root with either:

    python3 -m unittest discover -s tests -t .
    python3 -m pytest tests

Scope note: these originally covered ``research/`` only. They now also cover
the production path -- discovery, publication, logging redaction -- because
those carry the properties an operator relies on. Everything remains offline.

NETWORK KILL-SWITCH
-------------------
Importing this package disables outbound HTTP for the whole test process. No
test may talk to Metaculus or to a model provider: a test that reaches the
network could publish a forecast, burn provider quota, or make the suite's
result depend on someone else's uptime.

This is a backstop, not the primary control -- tests stub their own transports
-- but a backstop earned its place. While this suite was being written, a
module-caching bug in ``_real_forecasting_tools`` produced two distinct
``MetaculusClient`` classes; the fake transport was installed on one while
``publication.PublishingClient`` was bound to the other, and
``_post_question_prediction`` fell straight through to the real
implementation. It surfaced only as a mysterious hang inside
``_sleep_between_requests``. With this guard it would have surfaced instantly,
by name, as a blocked POST to Metaculus.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NetworkAccessDeniedInTests(RuntimeError):
    """A test tried to make a real network request."""


def _install_network_kill_switch() -> None:
    """Replace the transport primitives with something that raises.

    Patched at the lowest shared layer of each library rather than at every
    convenience wrapper, so `requests.post`, `requests.Session.post`,
    `urlopen` and friends are all covered by three patches.

    Deliberately NOT patched: `socket`. Tests legitimately construct
    `requests.Response` objects and the fixtures do local file I/O; blocking
    sockets outright breaks unrelated machinery for no extra safety.
    """
    try:
        import requests.adapters
    except Exception:  # pragma: no cover - requests is a hard dependency
        pass
    else:

        def denied_send(self, request, **kwargs):  # noqa: ANN001, ARG001
            raise NetworkAccessDeniedInTests(
                "a test attempted a real {0} to {1}. Tests must stub their "
                "transport; see tests/__init__.py".format(
                    request.method, request.url
                )
            )

        requests.adapters.HTTPAdapter.send = denied_send

    import urllib.request

    def denied_open(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        target = args[0] if args else "<unknown>"
        target = getattr(target, "full_url", target)
        raise NetworkAccessDeniedInTests(
            "a test attempted a real urlopen to {0}. Tests must stub their "
            "transport; see tests/__init__.py".format(target)
        )

    # test_read_api swaps `api.urllib.request.urlopen` per test and restores
    # it, so it overrides this for its own duration and puts this back after.
    urllib.request.urlopen = denied_open

    try:
        import http.client
    except Exception:  # pragma: no cover
        return

    def denied_connect(self):  # noqa: ANN001
        raise NetworkAccessDeniedInTests(
            "a test attempted a real connection to {0}:{1}. Tests must stub "
            "their transport; see tests/__init__.py".format(self.host, self.port)
        )

    http.client.HTTPConnection.connect = denied_connect
    http.client.HTTPSConnection.connect = denied_connect


_install_network_kill_switch()
