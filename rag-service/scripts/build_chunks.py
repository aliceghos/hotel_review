"""离线构建评论分块索引（优化2）

运行方式: cd rag-service && python scripts/build_chunks.py

将 filtered_comments.csv 中所有评论切分为句子级 chunk 并存入 ChromaDB。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from modules.clients import EmbeddingClient
from modules.chunking import ChunkIndexer

import os

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    print("构建评论分块索引...")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 缺少 DASHSCOPE_API_KEY")
        return

    df = pd.read_csv(DATA_DIR / "filtered_comments.csv", index_col=0)
    print(f"评论数: {len(df)}")

    ec = EmbeddingClient(api_key)
    ci = ChunkIndexer(DATA_DIR)
    ci.build(df, ec)
    print(f"✅ 完成，共 {ci.count()} 个分块")


if __name__ == "__main__":
    main()
