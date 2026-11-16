"""Resolution forensics (resolution_spec.py): one Haiku request with the web
fetch tool that pins down what resolves the question. No network: the
Anthropic client is replaced by a fake."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import resolution_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def usage(inp=1000, out=200, fetches=1):
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0,
              cache_read_input_tokens=0, server_tool_use=NS(web_search_requests=0,
                                                              web_fetch_requests=fetches))


def response(text="RESOLVES YES IF: x", stop="end_turn", fetches=1):
    return NS(content=[NS(type="text", text=text)] if text else [], stop_reason=stop,
              usage=usage(fetches=fetches), model="claude-haiku-5-5")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def run(client, **kw):
    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = lambda: client
    args = dict(question_text="Will X happen?", criteria="Resolves YES per https://example.org/data.",
                fine_print="See also https://www.metaculus.com/questions/1/ and https://b.gov/x",
                timing="Today is 2026-11-16.")
    args.update(kw)
    with mock.patch.dict(sys.modules, {"anthropic": fake}):
        return asyncio.run(resolution_spec.build(**args))


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"SPEND_LEDGER_PATH": os.path.join(self.dir.name, "l.jsonl")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.dir.cleanup()

    def test_the_spec_is_a_titled_section(self):
        out = run(FakeClient([response()]))
        self.assertTrue(out.startswith("## How this question resolves"))
        self.assertIn("RESOLVES YES IF: x", out)

    def test_the_request_is_haiku_with_a_capped_basic_fetch_tool(self):
        client = FakeClient([response()])
        run(client)
        call = client.calls[0]
        self.assertEqual(call["model"], "claude-haiku-5-5")
        self.assertEqual(call["tools"], [{"type": "web_fetch_20250910", "name": "web_fetch",
                                          "max_uses": resolution_spec.MAX_FETCHES,
                                          "max_content_tokens": resolution_spec.MAX_CONTENT_TOKENS}])
        # The links must be in the user message: the tool opens nothing else.
        user = call["messages"][0]["content"]
        self.assertIn("https://example.org/data", user)
        self.assertIn("https://b.gov/x", user)
        self.assertNotIn("metaculus.com/questions/1", user.split("Links in the criteria")[1])

    def test_a_paused_turn_is_resumed_append_only(self):
        client = FakeClient([response(text="Fetching the source.", stop="pause_turn"), response()])
        out = run(client)
        self.assertIn("RESOLVES YES IF", out)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["messages"][-1]["role"], "assistant")

    def test_failures_and_refusals_cost_only_the_spec(self):
        self.assertEqual(run(FakeClient([RuntimeError("down")])), "")
        self.assertEqual(run(FakeClient([response(stop="refusal")])), "")
        self.assertEqual(run(FakeClient([response(text="")])), "")

    def test_its_spend_is_recorded(self):
        with mock.patch.object(resolution_spec.spend, "record") as record:
            run(FakeClient([response()]))
        self.assertEqual(record.call_count, 1)
        self.assertEqual(record.call_args.kwargs["model"], "claude-haiku-5-5")
        self.assertGreater(record.call_args.kwargs["usd"], 0)

    def test_links_skip_metaculus_and_repeat_once(self):
        self.assertEqual(resolution_spec.links("a https://x.org/a. b https://x.org/a",
                                               "https://www.metaculus.com/q/1"),
                         ["https://x.org/a"])


class WiringTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            self.src = h.read()
        self.tail = self.src[self.src.index("opus_forecasts = spend.prize_forecasts()"):]

    def test_on_for_the_scored_forecasts_off_for_the_safety_net(self):
        for tour in ("futureeval", "minibench"):
            at = self.tail.index(f'spend.set_tournament("{tour}")')
            self.assertIn("template_bot.resolution_spec = True", self.tail[at:at + 300], tour)
        mp = self.tail.index('spend.set_tournament("market_pulse")')
        self.assertIn("template_bot.resolution_spec = True", self.tail[mp:mp + 300])
        early = self.tail.index('spend.set_tournament("market_pulse_early")')
        self.assertIn("template_bot.resolution_spec = False", self.tail[early:early + 300])

    def test_off_by_default_and_ahead_of_the_research(self):
        self.assertIn("    resolution_spec = False\n", self.src)
        body = self.src[self.src.index("    async def _run_research(self, question"):]
        body = body[:body.index("    async def _run_research_body")]
        self.assertIn('return f"{spec}\\n\\n{body}" if body else spec', body)


if __name__ == "__main__":
    unittest.main()
