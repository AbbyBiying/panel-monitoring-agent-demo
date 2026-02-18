from unittest.mock import AsyncMock, patch
import pathlib
import pytest


def _load_biz_chunks() -> list[dict]:
    """Read business_context.txt and split into chunks matching Firestore schema."""
    root = pathlib.Path(__file__).parent.parent.parent
    biz_path = root / "panel_monitoring" / "data" / "business_context.txt"

    if not biz_path.exists():
        print(f"\n[ERROR] File not found at: {biz_path}")
        return []

    print(f"\n[SUCCESS] Found biz_content at: {biz_path}")
    text = biz_path.read_text(encoding="utf-8")

    # Split on blank lines into sections (mirrors the ingest chunking).
    # Each chunk becomes a retrieved doc with a "text" key, same shape
    # that get_similar_patterns returns from Firestore.
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    return [{"text": section, "id": f"bctx_{i:03d}"} for i, section in enumerate(sections)]


# Dummy 768-dim vector so embed_text doesn't call the real embedding API
_DUMMY_VECTOR = [0.0] * 768


@pytest.fixture(autouse=True)
def use_local_prompts():
    biz_chunks = _load_biz_chunks()

    with patch(
        "panel_monitoring.app.nodes.get_active_prompt_spec",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "panel_monitoring.app.nodes.embed_text",
        new_callable=AsyncMock,
        return_value=_DUMMY_VECTOR,
    ), patch(
        "panel_monitoring.app.nodes.get_similar_patterns",
        new_callable=AsyncMock,
        return_value=biz_chunks,
    ):
        yield
