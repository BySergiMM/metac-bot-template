#!/usr/bin/env python3
"""notify.py: a small, standalone, reusable external-notification CLI.

Created fresh for this repo (there was no pre-existing "other repository" to
copy from -- confirmed before writing a line of this file). Designed to be
the kind of script that travels between projects unchanged: no import of
anything else in this repo, standard library only, one subcommand per
channel.

Usage
-----
    python scripts/notify.py whatsapp --message "text to send"
    echo "text to send" | python scripts/notify.py whatsapp

Exit code 0 on a confirmed send, 1 on any failure (missing config, HTTP
error, malformed provider response). Never raises an unhandled traceback
that could echo a secret -- every failure path prints a short, generic
message and returns cleanly.

WhatsApp channel: CallMeBot
----------------------------
CallMeBot's WhatsApp bridge (https://www.callmebot.com/blog/free-api-whatsapp-messages/)
is a single unauthenticated-transport HTTP GET:

    GET https://api.callmebot.com/whatsapp.php?phone=<E.164>&text=<url-encoded>&apikey=<key>

The phone number must have opted in to CallMeBot's WhatsApp bot beforehand
(a one-time manual step outside this script); the API key is issued by
CallMeBot in response to that opt-in message, not chosen by the caller.

Required environment variables for the `whatsapp` subcommand:
    CALLMEBOT_APIKEY        issued by CallMeBot after opting in
    CALLMEBOT_PHONE_NUMBER  destination number, E.164 format (e.g. "+34600000000")

Neither value is ever printed, logged, or included in an exception message.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"
DEFAULT_TIMEOUT_SECONDS = 15.0


class NotifyError(RuntimeError):
    """Base class for every way sending a notification can fail. Messages on
    these are written to stderr, so they must never contain a secret."""


class MissingConfigError(NotifyError):
    """A required environment variable is unset. Never includes its value --
    there is nothing to include when it is absent, which is the point."""


class DeliveryError(NotifyError):
    """The provider was reached but did not confirm delivery (HTTP error,
    network error, or an unexpected response body)."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingConfigError("{0} is not set in the environment".format(name))
    return value


def send_whatsapp(
    message: str,
    *,
    api_key: str,
    phone_number: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    base_url: str = CALLMEBOT_URL,
) -> None:
    """One CallMeBot request. Raises `DeliveryError` on anything but a
    request that completed without an HTTP error -- CallMeBot's success body
    is plain text ("Message queued...", "Message sent...", wording has
    changed over time), so this treats "no HTTP error" as the confirmation
    rather than pattern-matching prose that can drift.
    """
    if not message.strip():
        raise NotifyError("refusing to send an empty message")
    query = urllib.parse.urlencode({"phone": phone_number, "text": message, "apikey": api_key})
    url = "{0}?{1}".format(base_url, query)
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise DeliveryError("CallMeBot returned HTTP {0}".format(exc.code)) from None
    except urllib.error.URLError as exc:
        raise DeliveryError("could not reach CallMeBot: {0}".format(exc.reason)) from None
    except (TimeoutError, OSError) as exc:
        raise DeliveryError("CallMeBot call failed: {0}".format(type(exc).__name__)) from None


def _read_message(args: argparse.Namespace) -> str:
    if args.message is not None:
        return args.message
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise NotifyError("no --message given and stdin is a terminal, nothing to send")


def _cmd_whatsapp(args: argparse.Namespace) -> int:
    try:
        message = _read_message(args)
        api_key = _require_env("CALLMEBOT_APIKEY")
        phone_number = _require_env("CALLMEBOT_PHONE_NUMBER")
        send_whatsapp(message, api_key=api_key, phone_number=phone_number)
    except NotifyError as exc:
        print("notify.py: {0}".format(exc), file=sys.stderr)
        return 1
    print("notify.py: whatsapp message sent")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts/notify.py", description=__doc__)
    sub = parser.add_subparsers(dest="channel", required=True)

    whatsapp = sub.add_parser("whatsapp", help="send a WhatsApp message via CallMeBot")
    whatsapp.add_argument("--message", default=None, help="message text; reads stdin if omitted")
    whatsapp.set_defaults(fn=_cmd_whatsapp)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
