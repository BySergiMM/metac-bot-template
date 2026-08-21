"""R5: the research semaphore must survive main.py's two asyncio.run() calls.

main.py runs the tournament and MiniBench in two separate event loops
(main.py, `if __name__ == "__main__"`). asyncio primitives bind to the loop in
which a task first *waits* on them, and run_research holds the semaphore across
the LLM call, so with two or more questions the first loop always creates a
waiter. Re-entering the semaphore from the second loop then raised
"is bound to a different event loop", losing the entire MiniBench pass.

These tests exercise the REAL production attribute. They import the semaphore
through the same `self._concurrency_limiter` path run_research uses, on a
stand-in class carrying the production implementation, because importing
main.py itself requires forecasting-tools, which is not installed locally.
The property under test is copied by reference from main.py's source, so a
change there breaks these tests rather than silently bypassing them.
"""

from __future__ import annotations

import ast
import asyncio
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_production_limiter_class():
    """Build a class carrying main.py's REAL limiter implementation.

    The class body is extracted from main.py by AST, so this cannot drift: if
    someone rewrites the property, these tests run the rewritten version.
    """
    src = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = {
        "_max_concurrent_questions",
        "_concurrency_limiter_instance",
        "_concurrency_limiter_loop",
        "_concurrency_limiter",
    }
    pieces: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SummerTemplateBot2026":
            for item in node.body:
                names: set[str] = set()
                if isinstance(item, ast.Assign):
                    names = {t.id for t in item.targets if isinstance(t, ast.Name)}
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names = {item.target.id}
                elif isinstance(item, ast.FunctionDef):
                    names = {item.name}
                if names & wanted:
                    segment = ast.get_source_segment(src, item)
                    # get_source_segment starts at `def`, so decorators are not
                    # included. Dropping @property would silently turn the
                    # property into a bound method and make these tests
                    # exercise something production does not have.
                    decorators = getattr(item, "decorator_list", [])
                    if decorators:
                        lines = src.splitlines()
                        first = min(d.lineno for d in decorators) - 1
                        prefix = "\n".join(
                            line.strip() for line in lines[first:item.lineno - 1]
                        )
                        segment = prefix + "\n" + segment
                    pieces.append(segment)
            break
    assert pieces, "could not extract the limiter implementation from main.py"
    body = "\n".join("    " + line if line.strip() else line
                     for piece in pieces for line in piece.splitlines())
    namespace: dict = {"asyncio": asyncio}
    exec(compile("import asyncio\n\n\nclass Bot:\n" + body, "<main.py:limiter>", "exec"),
         namespace)
    return namespace["Bot"]


ProductionBot = load_production_limiter_class()


def fresh_bot():
    """A bot whose class state starts clean, mimicking a fresh process."""
    cls = type("Bot", (ProductionBot,), {
        "_concurrency_limiter_instance": None,
        "_concurrency_limiter_loop": None,
    })
    return cls()


async def process(bot, n_questions, tracker=None):
    """Mimic run_research: hold the semaphore across an await, N times."""
    async def one(_i):
        async with bot._concurrency_limiter:
            if tracker is not None:
                tracker["live"] += 1
                tracker["peak"] = max(tracker["peak"], tracker["live"])
            await asyncio.sleep(0)          # the LLM call yields here
            if tracker is not None:
                tracker["live"] -= 1
    await asyncio.gather(*[one(i) for i in range(n_questions)])


def in_new_loop(coro_factory):
    """One asyncio.run()-equivalent: a fresh loop, closed afterwards."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_factory())
    finally:
        loop.close()


class FirstLoopTests(unittest.TestCase):
    """Requirement 1."""

    def test_two_tasks_are_serialised_in_the_first_loop(self):
        bot = fresh_bot()
        tracker = {"live": 0, "peak": 0}
        in_new_loop(lambda: process(bot, 2, tracker))
        self.assertEqual(tracker["peak"], 1, "the semaphore must serialise")

    def test_sixty_tasks_are_serialised_in_the_first_loop(self):
        bot = fresh_bot()
        tracker = {"live": 0, "peak": 0}
        in_new_loop(lambda: process(bot, 60, tracker))
        self.assertEqual(tracker["peak"], 1)


class SecondLoopTests(unittest.TestCase):
    """Requirement 2: the regression itself."""

    def test_the_same_instance_works_in_a_second_loop(self):
        bot = fresh_bot()
        in_new_loop(lambda: process(bot, 2))      # binds the semaphore
        try:
            in_new_loop(lambda: process(bot, 60))  # would have raised
        except RuntimeError as exc:
            self.fail("second loop raised: {0}".format(exc))

    def test_no_bound_to_a_different_event_loop_error(self):
        bot = fresh_bot()
        in_new_loop(lambda: process(bot, 2))
        try:
            in_new_loop(lambda: process(bot, 60))
        except RuntimeError as exc:
            self.assertNotIn("bound to a different event loop", str(exc))
            raise

    def test_the_semaphore_object_is_replaced_between_loops(self):
        """Not merely reused: a loop-bound object cannot be carried over."""
        bot = fresh_bot()
        seen = []
        in_new_loop(lambda: _capture(bot, seen))
        in_new_loop(lambda: _capture(bot, seen))
        self.assertEqual(len(seen), 2)
        self.assertIsNot(seen[0], seen[1])


async def _capture(bot, seen):
    seen.append(bot._concurrency_limiter)


class ProductionSequenceTests(unittest.TestCase):
    """Requirements 3 and 4: exactly what main.py does."""

    def test_2_in_tournament_then_60_in_minibench(self):
        bot = fresh_bot()
        t = {"live": 0, "peak": 0}
        in_new_loop(lambda: process(bot, 2, t))    # asyncio.run(tournament)
        in_new_loop(lambda: process(bot, 60, t))   # asyncio.run(minibench)
        self.assertEqual(t["peak"], 1, "concurrency must still be 1 across both")

    def test_0_in_tournament_then_60_in_minibench(self):
        bot = fresh_bot()
        t = {"live": 0, "peak": 0}
        in_new_loop(lambda: process(bot, 60, t))
        self.assertEqual(t["peak"], 1)

    def test_60_then_60(self):
        bot = fresh_bot()
        t = {"live": 0, "peak": 0}
        in_new_loop(lambda: process(bot, 60, t))
        in_new_loop(lambda: process(bot, 60, t))
        self.assertEqual(t["peak"], 1)

    def test_three_consecutive_loops(self):
        bot = fresh_bot()
        for _ in range(3):
            in_new_loop(lambda: process(bot, 5))


class ConcurrencyIsStillOneTests(unittest.TestCase):
    """Requirement 5: the fix must not have widened the limit."""

    def test_the_limit_value_is_unchanged(self):
        self.assertEqual(ProductionBot._max_concurrent_questions, 1)

    def test_the_semaphore_is_built_with_that_limit(self):
        bot = fresh_bot()

        async def check():
            sem = bot._concurrency_limiter
            self.assertIsInstance(sem, asyncio.Semaphore)
            self.assertEqual(sem._value, 1)

        in_new_loop(check)

    def test_peak_concurrency_never_exceeds_one_under_load(self):
        bot = fresh_bot()
        t = {"live": 0, "peak": 0}
        in_new_loop(lambda: process(bot, 100, t))
        self.assertEqual(t["peak"], 1)

    def test_the_same_semaphore_is_shared_within_one_loop(self):
        """Rebuilding per loop must not become rebuilding per access, which
        would make the semaphore meaningless."""
        bot = fresh_bot()
        seen = []

        async def grab():
            for _ in range(5):
                seen.append(bot._concurrency_limiter)

        in_new_loop(grab)
        self.assertEqual(len({id(s) for s in seen}), 1)

    def test_two_instances_share_one_semaphore(self):
        """It was a class attribute; that must still hold."""
        cls = type("Bot", (ProductionBot,), {
            "_concurrency_limiter_instance": None,
            "_concurrency_limiter_loop": None,
        })
        a, b = cls(), cls()
        seen = []

        async def grab():
            seen.append(a._concurrency_limiter)
            seen.append(b._concurrency_limiter)

        in_new_loop(grab)
        self.assertIs(seen[0], seen[1])


class AccessOutsideALoopTests(unittest.TestCase):
    """The property must never replace a semaphore that may still be in use.

    Rebuilding is only safe when the previous loop is finished, which the
    per-loop branch guarantees. Reading the attribute with no loop running
    carries no such guarantee: the tasks holding it may belong to a loop that
    simply is not executing at this instant. Swapping the object there would
    let the next holder past a semaphore the current one no longer gates.
    """

    def test_access_outside_a_loop_returns_the_very_same_object(self):
        bot = fresh_bot()
        cls = type(bot)
        holder = {}

        async def take():
            semaphore = bot._concurrency_limiter
            await semaphore.acquire()          # genuinely in use
            holder["semaphore"] = semaphore

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(take())
            inside = holder["semaphore"]
            self.assertEqual(inside._value, 0, "the semaphore is held")

            # Step 4: read the property with no loop running at all.
            outside = bot._concurrency_limiter

            self.assertIs(outside, inside,
                          "the property handed back a different semaphore")
            self.assertIs(cls._concurrency_limiter_instance, inside,
                          "the class-level instance was replaced")
        finally:
            if "semaphore" in holder:
                holder["semaphore"].release()
            loop.close()

    def test_the_stored_loop_is_not_clobbered_either(self):
        """If the loop marker were reset, the next in-loop access would rebuild
        while the old semaphore is still held -- the same hazard, one step
        later."""
        bot = fresh_bot()
        cls = type(bot)
        holder = {}

        async def take():
            holder["semaphore"] = bot._concurrency_limiter
            await holder["semaphore"].acquire()
            holder["loop"] = asyncio.get_running_loop()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(take())
            bot._concurrency_limiter                     # read with no loop
            self.assertIs(cls._concurrency_limiter_loop, holder["loop"],
                          "the stored loop marker was overwritten")
        finally:
            if "semaphore" in holder:
                holder["semaphore"].release()
            loop.close()

    def test_repeated_outside_access_is_stable(self):
        bot = fresh_bot()
        holder = {}

        async def take():
            holder["semaphore"] = bot._concurrency_limiter
            await holder["semaphore"].acquire()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(take())
            seen = [bot._concurrency_limiter for _ in range(5)]
            self.assertEqual(len({id(s) for s in seen}), 1)
            self.assertIs(seen[0], holder["semaphore"])
        finally:
            holder["semaphore"].release()
            loop.close()

    def test_outside_access_with_nothing_built_yet_still_works(self):
        """The fall-through must survive: a first read outside a loop has
        nothing to preserve and must still hand back a usable semaphore."""
        bot = fresh_bot()
        semaphore = bot._concurrency_limiter
        self.assertIsInstance(semaphore, asyncio.Semaphore)
        self.assertEqual(semaphore._value, 1)

    def test_a_later_loop_still_gets_a_fresh_semaphore(self):
        """Preserving on the no-loop path must not disable per-loop rebuilding."""
        bot = fresh_bot()
        holder = {}

        async def take():
            holder["semaphore"] = bot._concurrency_limiter
            await holder["semaphore"].acquire()

        first = asyncio.new_event_loop()
        try:
            first.run_until_complete(take())
        finally:
            holder["semaphore"].release()
            first.close()

        bot._concurrency_limiter                          # read with no loop
        later = []
        in_new_loop(lambda: _capture(bot, later))
        self.assertIsNot(later[0], holder["semaphore"],
                         "a new loop must still get its own semaphore")


class NotepadLockTests(unittest.TestCase):
    """Requirement 6: does _note_pad_lock share the problem? No -- and why."""

    def test_a_lock_whose_body_never_awaits_does_not_bind(self):
        """ForecastBot._note_pad_lock guards three purely synchronous bodies
        (a list append, a list comprehension and an iteration). A lock that is
        never held across a yield point can never make a second task wait, so
        no waiter future is created and no loop binding occurs."""
        lock = asyncio.Lock()

        async def body(n):
            async def one(_i):
                async with lock:
                    pass          # no await: the SDK's critical sections
            await asyncio.gather(*[one(i) for i in range(n)])

        in_new_loop(lambda: body(2))
        in_new_loop(lambda: body(60))   # must not raise

    @unittest.skipIf(
        sys.version_info < (3, 10),
        "before 3.10 asyncio primitives bind at construction, so the binding "
        "is not caused by the yield point and this control proves nothing. "
        "CI runs 3.11, where it does run.",
    )
    def test_a_lock_whose_body_awaits_DOES_bind(self):
        """Control: the difference really is the yield point, which is why the
        semaphore needed fixing and the notepad lock did not."""
        lock = asyncio.Lock()

        async def body(n):
            async def one(_i):
                async with lock:
                    await asyncio.sleep(0)
            await asyncio.gather(*[one(i) for i in range(n)])

        in_new_loop(lambda: body(2))
        with self.assertRaises(RuntimeError):
            in_new_loop(lambda: body(60))


if __name__ == "__main__":
    unittest.main()
