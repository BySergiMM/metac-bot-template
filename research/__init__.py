"""Offline research lab for the seergiii-bot forecasting system.

Everything in this package is READ-ONLY with respect to Metaculus and has no
dependency on the production forecaster:

- it never imports ``main.py``;
- it never posts a forecast, a comment, or any other write request;
- it only ever touches data belonging to our own account, or questions that
  have already closed.

Design constraints, chosen deliberately:

1. **Standard library only.** The lab must run on a bare Python 3.9+ install
   with no virtualenv, so that measuring the bot never requires installing the
   bot. No numpy, no pandas, no requests.
2. **No shared state with production.** Adding a dependency here can never
   break a tournament run.
3. **Every number is traceable.** Any figure the lab reports must be
   attributable to a named dataset with a content hash.
"""

__version__ = "1.0.0"
