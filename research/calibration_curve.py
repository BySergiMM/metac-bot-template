"""The bot's calibration on resolved binaries, old system against current.

Metaculus' own curve on the profile is cumulative, so it cannot show whether
the current system (research.calibration.NEW_SYSTEM_START) is better calibrated
than the one before it. This prints calibration.binary_curve for every resolved
binary the bot forecast. Coarse bins of resolved questions only: aggregates.

Read only (MetaculusReadClient raises on any verb but GET); standard library
only, like the rest of research/.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import calibration  # noqa: E402
from research.metaculus_read_api import MetaculusReadClient, MetaculusReadError  # noqa: E402
from research.weekly_report import BOT_USER_ID, _questions  # noqa: E402


def main() -> int:
    client = MetaculusReadClient(os.environ.get("METACULUS_TOKEN", ""))
    questions, failed = [], 0
    for post in client.get_posts_forecasted_by(BOT_USER_ID):
        if not (post.get("resolved") or any(q.get("resolution") not in (None, "")
                                             for q in _questions(post))):
            continue
        # The listing has no my_forecasts: those come only from the post itself.
        try:
            questions.extend(_questions(client.get_json(f"/posts/{post['id']}/")))
        except MetaculusReadError:
            failed += 1
    print(f"calibration_curve questions={len(questions)} posts_failed={failed}")
    for system, row in calibration.binary_curve(questions).items():
        if not row["n"]:
            print(f"{system}: n=0")
            continue
        print(f"{system}: n={row['n']} share_yes={row['share_yes']:.3f} "
              f"mean_forecast={row['mean_forecast']:.3f} brier={row['brier']:.4f} "
              f"brier_of_the_share={row['brier_of_the_share']:.4f}")
        for b in row["bins"]:
            shown = (f"mean_forecast={b['mean_forecast']:.3f} share_yes={b['share_yes']:.3f}"
                     if b["n"] else "")
            print(f"  {b['from']:.1f}-{b['to']:.1f}: n={b['n']} {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
