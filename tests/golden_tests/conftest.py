from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def use_local_prompts():
    """
    Patch get_active_prompt_spec to return None so the graph
    always falls back to the local hardcoded prompts in prompts.py.
    """
    with patch(
        "panel_monitoring.app.nodes.get_active_prompt_spec",
        new_callable=AsyncMock,
        return_value=None,
    ):
        yield
