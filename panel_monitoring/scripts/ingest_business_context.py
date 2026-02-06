# panel_monitoring/scripts/ingest_business_context.py
"""
Ingest business_context.txt into Firestore fraud_patterns collection
following the LangChain Knowledge Base guide:
  1. Load with TextLoader
  2. Split with RecursiveCharacterTextSplitter
  3. Embed with VertexAIEmbeddings (text-embedding-004, 768-dim)
  4. Store in Firestore fraud_patterns collection

Usage:
  uv run python -m panel_monitoring panel_monitoring/scripts/ingest_business_context.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector
from langchain_community.document_loaders import TextLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from panel_monitoring.app.utils import load_credentials, make_credentials_from_env
from panel_monitoring.data.firestore_client import fraud_patterns_col

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path to business_context.txt relative to this script
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUSINESS_CONTEXT_PATH = DATA_DIR / "business_context.txt"

EMBEDDING_MODEL = "text-embedding-004"


def load_and_split() -> list:
    """Load business_context.txt and split into chunks using LangChain."""
    logger.info("Loading %s", BUSINESS_CONTEXT_PATH)
    loader = TextLoader(str(BUSINESS_CONTEXT_PATH))
    docs = loader.load()
    logger.info("Loaded %d document(s)", len(docs))
    
    # https://docs.langchain.com/oss/python/langchain/knowledge-base
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # We set add_start_index=True so that the character index where each split Document starts within the initial Document is preserved as metadata attribute “start_index”.
        add_start_index=True, 
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def get_embeddings_model() -> VertexAIEmbeddings:
    """Initialize VertexAIEmbeddings with proper credentials."""
    project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if os.getenv("ENVIRONMENT") == "local":
        creds = load_credentials()
    else:
        creds = make_credentials_from_env()

    return VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=project,
        location=location,
        credentials=creds,
    )


async def ingest():
    """Main ingestion pipeline: load → split → embed → store in Firestore."""
    # 1. Load and split
    chunks = load_and_split()

    # 2. Embed
    embeddings_model = get_embeddings_model()
    texts = [chunk.page_content for chunk in chunks]
    logger.info("Embedding %d chunks with %s...", len(texts), EMBEDDING_MODEL)
    vectors = embeddings_model.embed_documents(texts)
    logger.info("Generated %d embeddings (dim=%d)", len(vectors), len(vectors[0]))

    # 3. Store in Firestore
    col = await fraud_patterns_col()
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        doc_data = {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "source": "business_context.txt",
            "embedding": Vector(vector),
        }
        doc_ref = col.document(f"bctx_{i:03d}")
        await doc_ref.set(doc_data)
        logger.info("Stored chunk %d/%d: %s...", i + 1, len(chunks), chunk.page_content[:60])

    logger.info("Ingestion complete: %d chunks stored in fraud_patterns collection.", len(chunks))


if __name__ == "__main__":
    asyncio.run(ingest())
