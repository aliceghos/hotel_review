"""Elasticsearch BM25 text retrieval."""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from elasticsearch import Elasticsearch, helpers


class ElasticsearchBM25Index:
    """BM25 text search backed by Elasticsearch.

    The class replaces the local pickle-based inverted index in online retrieval.
    It can also bootstrap the index from the comment DataFrame at service startup.
    """

    def __init__(
        self,
        url: str,
        index_name: str = "hotel_comments",
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        verify_certs: bool = True,
        analyzer: str = "standard",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not url:
            raise ValueError("缺少 Elasticsearch 地址: ELASTICSEARCH_URL")

        self.index_name = index_name
        self.analyzer = analyzer
        self.k1 = k1
        self.b = b

        client_kwargs: dict[str, Any] = {
            "hosts": [url],
            "verify_certs": verify_certs,
            "request_timeout": 30,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        elif username and password:
            client_kwargs["basic_auth"] = (username, password)

        self.client = Elasticsearch(**client_kwargs)

    @classmethod
    def from_env(cls) -> "ElasticsearchBM25Index":
        """Create the index client from environment variables."""
        verify = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        return cls(
            url=os.getenv("ELASTICSEARCH_URL", ""),
            index_name=os.getenv("ELASTICSEARCH_INDEX", "hotel_comments"),
            api_key=os.getenv("ELASTICSEARCH_API_KEY") or None,
            username=os.getenv("ELASTICSEARCH_USERNAME") or None,
            password=os.getenv("ELASTICSEARCH_PASSWORD") or None,
            verify_certs=verify,
            analyzer=os.getenv("ELASTICSEARCH_ANALYZER", "standard"),
            k1=float(os.getenv("ELASTICSEARCH_BM25_K1", "1.5")),
            b=float(os.getenv("ELASTICSEARCH_BM25_B", "0.75")),
        )

    def ensure_index(self, df_comments: pd.DataFrame | None = None, auto_index: bool = True) -> None:
        """Ensure the target index exists and optionally populate it when empty."""
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=self._index_definition())

        if not auto_index or df_comments is None:
            return

        count = self.client.count(index=self.index_name).get("count", 0)
        if count == 0 and len(df_comments) > 0:
            self.bulk_index(df_comments)

    def bulk_index(self, df_comments: pd.DataFrame) -> None:
        """Bulk index comments from the project DataFrame."""
        actions = []
        for doc_id, row in df_comments.iterrows():
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": str(doc_id),
                    "_source": self._row_to_document(doc_id, row),
                }
            )

        if actions:
            helpers.bulk(self.client, actions, refresh=True)

    def search(
        self,
        query: str,
        topk: int = 10,
        room_type: str | None = None,
        fuzzy_room_type: str | None = None,
    ) -> list[tuple[str, float]]:
        """Search comments with Elasticsearch BM25 and metadata filters."""
        filters = []
        if room_type:
            filters.append({"term": {"room_type": room_type}})
        elif fuzzy_room_type:
            filters.append({"term": {"fuzzy_room_type": fuzzy_room_type}})

        body = {
            "size": topk,
            "_source": False,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "comment^3",
                                    "category1",
                                    "category2",
                                    "category3",
                                    "room_type",
                                    "fuzzy_room_type",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": filters,
                }
            },
        }

        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [(hit["_id"], float(hit.get("_score", 0.0))) for hit in hits]

    def _index_definition(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "k1": self.k1,
                            "b": self.b,
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "comment": {"type": "text", "analyzer": self.analyzer},
                    "score": {"type": "float"},
                    "publish_date": {"type": "date", "ignore_malformed": True},
                    "quality_score": {"type": "float"},
                    "review_count": {"type": "integer"},
                    "useful_count": {"type": "integer"},
                    "room_type": {"type": "keyword"},
                    "fuzzy_room_type": {"type": "keyword"},
                    "category1": {"type": "keyword"},
                    "category2": {"type": "keyword"},
                    "category3": {"type": "keyword"},
                }
            },
        }

    def _row_to_document(self, doc_id: Any, row: pd.Series) -> dict[str, Any]:
        return {
            "comment_id": str(doc_id),
            "comment": self._clean_value(row.get("comment", "")),
            "score": self._clean_number(row.get("score", 0)),
            "publish_date": self._clean_date(row.get("publish_date")),
            "quality_score": self._clean_number(row.get("quality_score", 0)),
            "review_count": int(self._clean_number(row.get("review_count", 0))),
            "useful_count": int(self._clean_number(row.get("useful_count", 0))),
            "room_type": self._clean_value(row.get("room_type", "")),
            "fuzzy_room_type": self._clean_value(row.get("fuzzy_room_type", "")),
            "category1": self._clean_value(row.get("category1", "")),
            "category2": self._clean_value(row.get("category2", "")),
            "category3": self._clean_value(row.get("category3", "")),
        }

    def _clean_value(self, value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value)

    def _clean_number(self, value: Any) -> float:
        if pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _clean_date(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value)
