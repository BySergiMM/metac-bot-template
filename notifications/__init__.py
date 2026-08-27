"""WhatsApp status notifications for the production bot.

Four pieces, kept separate on purpose (see the report that accompanied this
package for the full design rationale):

    events.py        event detection -- what changed, from data main.py
                      already computes this run
    state.py          the one thing that must survive between runs: which
                      WhatsApp events have already been sent
    dashboard.py       builds the short message text
    integration.py     `main.py`'s single call site, `handle_run(...)`

`scripts/notify.py` (a sibling, not a submodule of this package) is the
actual CallMeBot client -- generic, reusable, no dependency on this package
or on `main.py`.

Nothing here imports `forecasting_tools`, touches `research/replay`,
`main.py`'s prompts/models, or Metaculus directly.
"""

from __future__ import annotations
