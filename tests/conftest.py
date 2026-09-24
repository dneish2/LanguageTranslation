"""Suite-wide defaults.

Tests run as a Cloud deployment unless they opt into Local. Local-first routing
probes a real Ollama on this machine, so without this default the same test
passes or fails depending on whether a developer happens to have Ollama running
(the dev box does, CI does not). A test about local routing says so with
``@pytest.mark.local_deployment``, or replaces ``_live_local_provider`` outright.
"""
import pytest

import TranslationBackend as tb


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local_deployment: run as a Local deployment (local-first routing on)")


@pytest.fixture(autouse=True)
def _deployment_mode(request, monkeypatch):
    local = request.node.get_closest_marker("local_deployment") is not None
    monkeypatch.setattr(tb, "DEPLOYMENT_MODE", "local" if local else "cloud")
