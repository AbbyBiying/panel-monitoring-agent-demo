# panel_monitoring/data/firestore_client.py
from __future__ import annotations

import asyncio
import os
import logging
from typing import Optional

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from google.cloud.firestore_v1.base_query import FieldFilter
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from panel_monitoring.app.utils import (
    load_credentials,
    log_info,
    make_credentials_from_env,
)
from panel_monitoring.app.retry import embedding_retry, firestore_retry
from panel_monitoring.models.firestore_docs import PromptSpecDoc

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-004"
_EMBEDDINGS: Optional[GoogleGenerativeAIEmbeddings] = None

_DB: Optional[AsyncClient] = None
_DB_LOCK = asyncio.Lock()


async def get_db() -> AsyncClient:
    """
    Return a singleton async Firestore client with support for emulator and non-default DB.

    Environment variables:
        GCP_PROJECT or GOOGLE_CLOUD_PROJECT - Firestore project ID
        FIRESTORE_DATABASE_ID - Optional, defaults to "(default)"
        FIRESTORE_EMULATOR_HOST - Optional, if using emulator
        GOOGLE_APPLICATION_CREDENTIALS - Path to credentials file
    """
    global _DB
    if _DB is not None:
        return _DB

    async with _DB_LOCK:
        # Double-check after acquiring lock
        if _DB is not None:
            return _DB

        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        database = os.getenv("FIRESTORE_DATABASE_ID", "panel-monitoring-agent-dev")

        # Load credentials in a thread to avoid blocking
        if os.getenv("ENVIRONMENT") == "local":
            logger.info("Running in LOCAL environment, loading credentials from file.")
            log_info("Running in local environment, loading credentials from file.")
            creds = await asyncio.to_thread(load_credentials)
        else:
            log_info("Running in NOT LOCAL environment, loading credentials from Path.")
            creds = await asyncio.to_thread(make_credentials_from_env)
            logger.info(
                "Running in NOT LOCAL environment, loading credentials from Path."
            )
        logger.debug("creds type: %s", type(creds))

        try:
            _DB = AsyncClient(project=project, database=database, credentials=creds)
            logger.info(
                "Initialized async Firestore client for project '%s' (DB: %s)",
                project,
                database,
            )
        except DefaultCredentialsError as e:
            raise RuntimeError(
                "Firestore credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS "
                "and GCP_PROJECT/GOOGLE_CLOUD_PROJECT, or run `gcloud auth application-default login`, "
                "or set FIRESTORE_EMULATOR_HOST."
            ) from e

        return _DB


async def events_col() -> AsyncCollectionReference:
    """Return reference to top-level 'events' collection. Each doc includes `project_id`."""
    try:
        db = await get_db()
        return db.collection("events")
    except Exception:
        logger.exception("Failed to get 'events' collection")
        raise


async def runs_col() -> AsyncCollectionReference:
    """Return reference to top-level 'runs' collection. Each doc includes `project_id` and `event_id`."""
    try:
        db = await get_db()
        return db.collection("runs")
    except Exception:
        logger.exception("Failed to get 'runs' collection")
        raise


async def alerts_col() -> AsyncCollectionReference:
    """Return reference to top-level 'alerts' collection. Each doc includes `project_id`."""
    try:
        db = await get_db()
        return db.collection("alerts")
    except Exception:
        logger.exception("Failed to get 'alerts' collection")
        raise


async def projects_col() -> AsyncCollectionReference:
    """Return reference to flat 'projects' collection for metadata."""
    try:
        db = await get_db()
        return db.collection("projects")
    except Exception:
        logger.exception("Failed to get 'projects' collection")
        raise


async def prompt_specs_col() -> AsyncCollectionReference:
    """Return reference to top-level 'prompt_specs' collection."""
    try:
        db = await get_db()
        return db.collection("prompt_specs")
    except Exception:
        logger.exception("Failed to get 'prompt_specs' collection")
        raise


async def get_active_prompt_spec(role: str) -> Optional[PromptSpecDoc]:
    """Query for a live PromptSpec with the given deployment_role. Returns None if not found."""
    try:
        return await _get_active_prompt_spec_inner(role)
    except Exception:
        logger.exception("get_active_prompt_spec failed for role=%s", role)
        return None


@firestore_retry
async def _get_active_prompt_spec_inner(role: str) -> Optional[PromptSpecDoc]:
    """Inner function with retry — raises on failure so tenacity can retry."""
    col = await prompt_specs_col()
    query = (
        col.where(filter=FieldFilter("deployment_status", "==", "live"))
        .where(filter=FieldFilter("deployment_role", "==", role))
        .limit(1)
    )
    docs = [doc async for doc in query.stream()]
    if not docs:
        return None
    data = docs[0].to_dict()
    spec = PromptSpecDoc.model_validate(data)
    spec.doc_id = docs[0].id
    return spec


async def fraud_patterns_col() -> AsyncCollectionReference:
    """Return reference to top-level 'fraud_patterns' collection for vector search."""
    try:
        db = await get_db()
        return db.collection("fraud_patterns")
    except Exception:
        logger.exception("Failed to get 'fraud_patterns' collection")
        raise


def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Return a singleton GoogleGenerativeAIEmbeddings instance."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    try:
        if os.getenv("ENVIRONMENT") == "local":
            creds = load_credentials()
        else:
            creds = make_credentials_from_env()

        # google-genai SDK requires scoped credentials to refresh tokens
        # This tells the service account "you're authorized for the cloud-platform scope" so when google-genai refreshes the token,
        # the OAuth server accepts it. Same credentials, same permissions, just explicitly labeled now.
        if creds and not creds.scopes:
            creds = creds.with_scopes(
                ["https://www.googleapis.com/auth/cloud-platform"]
            )

        _EMBEDDINGS = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            project=project,
            location=location,
            credentials=creds,
        )
    except Exception:
        logger.exception(
            "Failed to initialize GoogleGenerativeAIEmbeddings (project=%s, location=%s)",
            project,
            location,
        )
        raise
    return _EMBEDDINGS


@embedding_retry
def _embed_text_sync(text: str) -> list[float]:
    """Synchronous embed: init model + call API (all blocking I/O)."""
    try:
        model = _get_embeddings_model()
        vectors = model.embed_documents([text])
        logger.debug(
            "_embed_text_sync: embedded text (%d chars) -> %d-dim vector",
            len(text),
            len(vectors[0]),
        )
        return vectors[0]
    except Exception:
        logger.exception("_embed_text_sync: failed to embed text (%d chars)", len(text))
        raise


async def embed_text(text: str) -> list[float]:
    """Embed a single text string using Vertex AI text-embedding-004 (768-dim)."""
    try:
        return await asyncio.to_thread(_embed_text_sync, text)
    except Exception:
        logger.exception("embed_text: failed for text (%d chars)", len(text))
        raise


@firestore_retry
async def get_similar_patterns(query_vector: list[float], limit: int = 3) -> list[dict]:
    """
    Search for similar fraud patterns using vector similarity.
    Dimensions: 768 (Vertex AI text-embedding-004)
    """
    try:
        col = await fraud_patterns_col()

        # The actual vector search query
        vector_query = col.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=limit,
        )

        results = []
        async for doc in vector_query.stream():
            data = doc.to_dict()
            # Clean up: don't pass the raw embedding to the LLM
            data.pop("embedding", None)
            # Include the doc ID so the LLM can reference 'Rule XYZ'
            data["id"] = doc.id
            results.append(data)

        logger.debug(
            "get_similar_patterns: found %d results (limit=%d)", len(results), limit
        )
        return results
    except Exception:
        logger.exception("get_similar_patterns: vector search failed (limit=%d)", limit)
        raise
