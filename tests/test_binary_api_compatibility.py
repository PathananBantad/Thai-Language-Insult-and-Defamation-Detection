import importlib.util

import pytest


def test_legacy_api_routes_remain_and_binary_route_is_separate():
    if not importlib.util.find_spec("fastapi"):
        pytest.skip("FastAPI is not installed in the specified interpreter")
    from src.thai_mod_api.main import app

    paths = {route.path for route in app.routes}
    assert "/api/predict" in paths
    assert "/api/batch-predict" in paths
    assert "/api/binary/insult-defamation/predict" in paths
