"""Vector retrieval engine wrappers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import dashvector


class VectorCollection(Protocol):
    """Minimal collection contract used by the hybrid retriever."""

    def query(self, *args, **kwargs) -> Any:
        ...


@dataclass
class DashVectorSearchEngine:
    """DashVector-backed retrieval engine for comment and reverse-query vectors."""

    api_key: str
    endpoint: str
    comment_collection_name: str = "comment_database"
    reverse_collection_name: str = "reverse_query_database"

    def __post_init__(self) -> None:
        self.client = dashvector.Client(api_key=self.api_key, endpoint=self.endpoint)
        self.comments = self.client.get(self.comment_collection_name)
        self.reverse_queries = self.client.get(self.reverse_collection_name)

    @classmethod
    def from_env(cls) -> "DashVectorSearchEngine":
        api_key = os.getenv("DASHVECTOR_API_KEY")
        endpoint = os.getenv("DASHVECTOR_HOTEL_ENDPOINT")
        if not api_key or not endpoint:
            raise ValueError("缺少 DashVector 配置: DASHVECTOR_API_KEY / DASHVECTOR_HOTEL_ENDPOINT")
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            comment_collection_name=os.getenv("DASHVECTOR_COMMENT_COLLECTION", "comment_database"),
            reverse_collection_name=os.getenv("DASHVECTOR_REVERSE_COLLECTION", "reverse_query_database"),
        )


def create_vector_search_engine(
    engine_name: str,
    dashvector_api_key: str,
    dashvector_endpoint: str,
) -> DashVectorSearchEngine:
    """Create a vector search engine.

    DashVector is the runtime engine for this project. The factory keeps the
    rest of the RAG pipeline independent from a concrete vector-store client.
    """
    normalized = engine_name.lower().strip()
    if normalized != "dashvector":
        raise ValueError(f"暂不支持的向量检索引擎: {engine_name}")

    return DashVectorSearchEngine(
        api_key=dashvector_api_key,
        endpoint=dashvector_endpoint,
        comment_collection_name=os.getenv("DASHVECTOR_COMMENT_COLLECTION", "comment_database"),
        reverse_collection_name=os.getenv("DASHVECTOR_REVERSE_COLLECTION", "reverse_query_database"),
    )
