"""评论文本分块（优化2）：句子级切分 + ChromaDB 存储 + 向量检索"""

import re
from pathlib import Path
from typing import Optional

import chromadb


class CommentChunker:
    """评论文本分块器：按句子边界切分，1-2句/块，相邻块重叠1句"""

    _SENT_SPLIT = re.compile(r'(?<=[。！？!?\n])(?=[^\s])')

    def __init__(self, max_chunk_sentences: int = 2, overlap_sentences: int = 1):
        self.max_sentences = max_chunk_sentences
        self.overlap = overlap_sentences

    def chunk(self, text: str) -> list[dict]:
        """将文本切成块，返回 [{text, start_char, end_char, chunk_idx}, ...]"""
        if not text or not text.strip():
            return []

        # 按句子边界拆分
        raw_sentences = self._SENT_SPLIT.split(text)

        # 过滤空句，记录每个句子的起止位置
        sentences: list[dict] = []
        pos = 0
        for s in raw_sentences:
            stripped = s.strip()
            if not stripped:
                pos = text.find(s, pos) + len(s) if s else pos
                continue
            start = text.find(stripped, pos) if stripped in text[pos:] else pos
            if start < pos:
                start = pos
            end = start + len(stripped)
            sentences.append({"text": stripped, "start": start, "end": end})
            pos = end

        if not sentences:
            return [{"text": text.strip(), "start_char": 0, "end_char": len(text), "chunk_idx": 0}]

        # 分组为 chunk
        chunks = []
        i = 0
        while i < len(sentences):
            group = sentences[i:i + self.max_sentences]
            chunk_text = "".join(s["text"] for s in group)
            chunks.append({
                "text": chunk_text,
                "start_char": group[0]["start"],
                "end_char": group[-1]["end"],
                "chunk_idx": len(chunks),
            })
            # 前进步长 = max_sentences - overlap
            i += max(1, self.max_sentences - self.overlap)

        return chunks


class ChunkIndexer:
    """评论分块索引：ChromaDB 存储 + 向量检索

    每个 chunk 存储为 ChromaDB 文档，metadata 包含 comment_id / chunk_idx / start_char / end_char。
    """

    COLLECTION_NAME = "chunk_database"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        chroma_client = chromadb.PersistentClient(path=str(data_dir / "chroma_db"))
        try:
            self.collection = chroma_client.get_collection(self.COLLECTION_NAME)
        except Exception:
            self.collection = chroma_client.create_collection(self.COLLECTION_NAME)
        self._chunker = CommentChunker()

    def build(self, df_comments, embedding_client, batch_size: int = 10):
        """从评论 DataFrame 构建分块索引（增量：跳过已有 comment_id）"""
        existing_ids = set()
        try:
            # 获取已有 chunk 的 comment_id（去重）
            all_meta = self.collection.get(include=["metadatas"])
            if all_meta["metadatas"]:
                for m in all_meta["metadatas"]:
                    existing_ids.add(m.get("comment_id", ""))
        except Exception:
            pass

        new_count = 0
        for idx in df_comments.index:
            cid = str(idx)
            if cid in existing_ids:
                continue
            comment_text = str(df_comments.loc[idx, "comment"])
            chunks = self._chunker.chunk(comment_text)
            if not chunks:
                continue

            for c in chunks:
                chunk_id = f"{cid}_{c['chunk_idx']}"
                # batch embed
                emb = embedding_client.embed_batch([c["text"]])[0]
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[emb],
                    documents=[c["text"]],
                    metadatas=[{
                        "comment_id": cid,
                        "chunk_idx": c["chunk_idx"],
                        "start_char": c["start_char"],
                        "end_char": c["end_char"],
                    }],
                )
                new_count += 1

        print(f"ChunkIndexer: 新增 {new_count} 个分块，共计 {self.collection.count()} 个")

    def search(self, query_embedding: list[float], topk: int = 30) -> list[dict]:
        """向量检索分块，返回 [{comment_id, chunk_idx, text, start_char, end_char}, ...]"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=topk,
            )
        except Exception:
            return []

        entries = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i] if results["metadatas"][0] else {}
                entries.append({
                    "comment_id": meta.get("comment_id", ""),
                    "chunk_idx": meta.get("chunk_idx", 0),
                    "text": results["documents"][0][i] if results["documents"][0] else "",
                    "start_char": meta.get("start_char", 0),
                    "end_char": meta.get("end_char", 0),
                    "distance": results["distances"][0][i] if results["distances"] and results["distances"][0] else 0.0,
                })
        return entries

    def build_if_empty(self, df_comments, embedding_client):
        """如果索引为空，小数据量时自动构建；大数据量时提示离线构建"""
        if self.collection.count() == 0:
            if len(df_comments) <= 100:
                print("ChunkIndexer: 索引为空，开始构建...")
                self.build(df_comments, embedding_client)
            else:
                print(f"ChunkIndexer: 索引为空且评论数较多({len(df_comments)}条)，"
                      f"跳过自动构建。请运行 scripts/build_chunks.py 离线构建分块索引。")

    def count(self) -> int:
        return self.collection.count()
