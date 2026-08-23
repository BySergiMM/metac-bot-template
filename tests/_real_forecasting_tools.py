"""Load the REAL forecasting_tools past the suite's stubs, without disturbing them.

Five test modules (test_alibaba_provider, test_balanced_llm, test_fallback_llm,
test_fallback_e2e_simulation, test_rate_limiter) install a minimal stub into
``sys.modules["forecasting_tools"]`` so they can run on a machine where the
package is not installed. Each guards on ``if "forecasting_tools" not in
sys.modules``, so the first of them to import wins for the whole process.

Alphabetically ``test_alibaba_provider`` sorts first, so by the time
``test_discovery`` and ``test_publication`` import, ``forecasting_tools`` is a
bare ``ModuleType`` with no ``__path__`` and ``import
forecasting_tools.helpers.metaculus_client`` fails with "not a package".

``publication`` and ``discovery`` are production modules that genuinely need
the real client, so the fix belongs here rather than in either place:

* the stub entries are lifted out of ``sys.modules``,
* the real package is imported,
* the stub entries are put back exactly as they were.

Putting them back matters. The stub modules already hold direct references to
their stub classes, so they are unaffected either way -- but a stub module
importing LATER checks ``sys.modules`` and would skip stubbing if it found the
real package, silently changing what it tests. Restoring keeps every other
module's view of the world byte-identical to a run without this helper.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

_PREFIX = "forecasting_tools"


def _is_stub(module: object) -> bool:
    """A real package has __path__; the suite's stand-in is a bare module."""
    return module is not None and not hasattr(module, "__path__")


#: The real forecasting_tools modules, imported ONCE and reused forever.
#:
#: This cache is the whole safety property. An earlier version deleted the
#: stub entries and re-imported the package on every entry, which produced a
#: SECOND set of module objects -- and therefore a second MetaculusClient
#: CLASS. tests/test_publication then patched the network methods onto one
#: class while publication.PublishingClient was bound to the other, so
#: `super()._post_question_prediction(...)` reached the REAL implementation and
#: the suite began issuing live HTTP requests to Metaculus. It surfaced only
#: as a hang inside `_sleep_between_requests`.
#:
#: Importing once and swapping the SAME objects back in guarantees exactly one
#: MetaculusClient class in the process, which is what makes patching it
#: meaningful.
_REAL_MODULES: dict = {}


def _capture_real_modules() -> None:
    """Import the real package once and remember every module it created."""
    if _REAL_MODULES:
        return
    stubbed = _current_forecasting_tools_modules()
    for name in stubbed:
        del sys.modules[name]
    try:
        # Importing the client pulls in the data models and helpers this suite
        # needs; anything else is imported lazily against these same modules.
        importlib.import_module(_PREFIX + ".helpers.metaculus_client")
        _REAL_MODULES.update(_current_forecasting_tools_modules())
    finally:
        for name in _current_forecasting_tools_modules():
            del sys.modules[name]
        sys.modules.update(stubbed)


def _current_forecasting_tools_modules() -> dict:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == _PREFIX or name.startswith(_PREFIX + ".")
    }


@contextlib.contextmanager
def real_forecasting_tools():
    """Make the real package importable for the duration of the block.

    A context manager rather than a plain loader because the modules that
    NEED it -- ``publication`` and ``discovery`` -- do their own
    ``from forecasting_tools.helpers... import ...`` at import time, and
    ``BinaryReport.publish_report_to_metaculus`` re-imports at CALL time.
    Returning the real modules and then restoring the stubs would put the stub
    back before those imports ever ran.
    """
    if not _is_stub(sys.modules.get(_PREFIX)):
        # The real package is already the live one; nothing to swap.
        yield
        return

    _capture_real_modules()
    stubbed = _current_forecasting_tools_modules()
    for name in stubbed:
        del sys.modules[name]
    sys.modules.update(_REAL_MODULES)
    try:
        yield
    finally:
        # Remove ONLY what we put in, then restore the stubs, so the next
        # stub-installing module sees exactly what it expected. The real
        # module OBJECTS survive in _REAL_MODULES and are reused next time.
        for name in list(_REAL_MODULES):
            sys.modules.pop(name, None)
        for name in _current_forecasting_tools_modules():
            del sys.modules[name]
        sys.modules.update(stubbed)


def load(*module_names: str) -> tuple:
    """Return the named real forecasting_tools submodules."""
    with real_forecasting_tools():
        return tuple(importlib.import_module(name) for name in module_names)


def the_real_metaculus_client():
    """The one and only real MetaculusClient class in this process."""
    (module,) = load(_PREFIX + ".helpers.metaculus_client")
    return module.MetaculusClient
