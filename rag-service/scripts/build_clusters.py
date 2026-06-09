"""构建聚类类别系统（优化4）

运行方式: cd rag-service && python scripts/build_clusters.py

流程：
1. 加载评论数据并生成 embeddings
2. KMeans 聚类 + 软多标签分配
3. LLM 生成聚类标签
4. LLM 生成聚类摘要
5. 更新 ChromaDB 摘要库
"""

import sys
import os
import json
import time
from pathlib import Path

# 确保从 rag-service 目录运行
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import chromadb
from dotenv import load_dotenv
load_dotenv()

from modules.clients import LLMClient, EmbeddingClient
from modules.clustering import (
    CommentClustering, generate_cluster_labels, build_cluster_summaries
)

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    print("=" * 60)
    print("优化4：聚类类别系统构建")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载评论数据...")
    csv_path = DATA_DIR / "filtered_comments.csv"
    df = pd.read_csv(csv_path, index_col=0)
    comments = df["comment"].tolist()
    print(f"  加载 {len(comments)} 条评论")

    # 2. 生成 embeddings
    print("\n[2/6] 生成评论 embeddings...")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("  ❌ 缺少 DASHSCOPE_API_KEY")
        return

    embed_client = EmbeddingClient(api_key)
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        embs = embed_client.embed_batch(batch)
        all_embeddings.extend(embs)
        if (i // batch_size) % 50 == 0:
            print(f"  {i}/{len(comments)} ...")
    embeddings = np.array(all_embeddings)
    print(f"  ✅ 生成 {len(embeddings)} 个 embedding，维度 {embeddings.shape[1]}")

    # 3. KMeans 聚类
    print("\n[3/6] KMeans 聚类 + 软多标签分配...")
    clustering = CommentClustering(n_clusters=14, random_state=42)
    clustering.fit(embeddings)
    hard_labels = clustering.labels_

    # 统计每簇大小
    unique, counts = np.unique(hard_labels, return_counts=True)
    for cid, cnt in zip(unique, counts):
        print(f"  聚类 {cid}: {cnt} 条评论")

    # 软多标签分配
    soft_labels = clustering.assign_soft(embeddings, top_k=3)

    # 保存聚类模型
    model_path = DATA_DIR / "cluster_model.json"
    clustering.save(model_path)

    # 保存软标签
    soft_path = DATA_DIR / "cluster_assignments.json"
    soft_records = []
    for i, labels in enumerate(soft_labels):
        for label in labels:
            soft_records.append({
                "comment_index": i,
                "comment_id": str(df.index[i]),
                "cluster": label["cluster"],
                "score": label["score"],
            })
    with open(soft_path, 'w', encoding='utf-8') as f:
        json.dump(soft_records, f, ensure_ascii=False)
    print(f"  软标签已保存: {soft_path}")

    # 4. LLM 生成聚类标签
    print("\n[4/6] LLM 生成聚类标签...")
    llm_client = LLMClient(api_key, model="qwen-flash")
    cluster_texts = {}
    for i, label in enumerate(hard_labels):
        cid = int(label)
        if cid not in cluster_texts:
            cluster_texts[cid] = []
        cluster_texts[cid].append(comments[i])

    cluster_labels = generate_cluster_labels(llm_client, cluster_texts)
    labels_path = DATA_DIR / "cluster_labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_labels, f, ensure_ascii=False)
    print(f"  标签已保存: {labels_path}")

    # 5. LLM 生成聚类摘要
    print("\n[5/6] LLM 生成聚类摘要...")
    summaries = build_cluster_summaries(llm_client, cluster_texts, cluster_labels)
    summaries_path = DATA_DIR / "cluster_summaries.json"
    with open(summaries_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"  摘要已保存: {summaries_path}")

    # 6. 更新 ChromaDB
    print("\n[6/6] 更新 ChromaDB 摘要库...")
    chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))

    # 删除旧摘要库
    try:
        chroma_client.delete_collection("summary_database")
    except Exception:
        pass

    # 创建新摘要库
    COLLECTION_NAME = "summary_database"
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = chroma_client.create_collection(COLLECTION_NAME)

    for i, s in enumerate(summaries):
        emb = embed_client.embed_batch([s["keywords"]])[0]
        collection.add(
            ids=[f"summary_{i}"],
            embeddings=[emb],
            documents=[s["summary"]],
            metadatas=[{
                "category": s["category"],
                "keywords": s["keywords"],
                "comment_count": s["comment_count"],
            }],
        )

    print(f"  ✅ 摘要库已更新，共 {collection.count()} 条")

    # 完成
    print("\n" + "=" * 60)
    print("✅ 聚类类别系统构建完成！")
    print(f"  - 聚类数: {clustering.n_clusters}")
    print(f"  - 聚类标签: {list(cluster_labels.values())}")
    print(f"  - 摘要文件: {summaries_path}")
    print(f"  - ChromaDB: {collection.count()} 条摘要")
    print("=" * 60)


if __name__ == "__main__":
    main()
