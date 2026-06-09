"""评论聚类与多标签分配（优化4）：KMeans 聚类 + 软多标签"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


class CommentClustering:
    """评论聚类器：KMeans + 软多标签分配

    使用 comment embeddings 进行 KMeans 聚类，然后为每条评论计算到所有聚类中心
    的余弦相似度，取 top-K 作为软标签。
    """

    def __init__(self, n_clusters: int = 14, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """拟合 KMeans 并返回硬标签"""
        embeddings = np.asarray(embeddings, dtype=float)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.labels_ = self.kmeans.fit_predict(embeddings)
        self.cluster_centers_ = normalize(self.kmeans.cluster_centers_)
        return self.labels_

    def select_best_k(self, embeddings: np.ndarray, k_range: range = range(10, 21)) -> int:
        """通过轮廓系数选择最佳 K 值"""
        embeddings = np.asarray(embeddings, dtype=float)
        best_k = k_range.start
        best_score = -1
        for k in k_range:
            if k >= len(embeddings):
                break
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        self.n_clusters = best_k
        return best_k

    def assign_soft(self, embeddings: np.ndarray, top_k: int = 3) -> list[list[dict]]:
        """软多标签分配：每条评论返回 top-K 聚类及余弦相似度得分

        返回:
            [[{"cluster": 0, "score": 0.85}, {"cluster": 3, "score": 0.62}, ...], ...]
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("请先调用 fit() 训练聚类模型")

        embeddings = np.asarray(embeddings, dtype=float)
        embeddings_norm = normalize(embeddings)
        # 余弦相似度 = 归一化向量的点积
        sim_matrix = embeddings_norm @ self.cluster_centers_.T

        results = []
        for i in range(len(embeddings)):
            scores = sim_matrix[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            # 归一化 top-K 得分，使之和为 1
            top_scores = scores[top_indices]
            top_scores = np.maximum(top_scores, 0)  # 截断负值
            if top_scores.sum() > 0:
                top_scores = top_scores / top_scores.sum()
            entries = [
                {"cluster": int(idx), "score": round(float(score), 4)}
                for idx, score in zip(top_indices, top_scores)
            ]
            results.append(entries)

        return results

    def save(self, filepath: Path):
        """保存聚类模型（仅中心点，无需 sklearn）"""
        data = {
            "n_clusters": self.n_clusters,
            "cluster_centers": self.cluster_centers_.tolist() if self.cluster_centers_ is not None else [],
            "labels": self.labels_.tolist() if self.labels_ is not None else [],
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"聚类模型已保存: {filepath}")

    def load(self, filepath: Path):
        """加载聚类中心点"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.n_clusters = data["n_clusters"]
        self.cluster_centers_ = np.array(data["cluster_centers"])
        self.labels_ = np.array(data["labels"]) if data["labels"] else None
        print(f"聚类模型已加载: {filepath}")

    def assign_soft_from_centers(self, embeddings: np.ndarray, top_k: int = 3) -> list[list[dict]]:
        """使用加载的中心点进行软分配（无需重新拟合）"""
        if self.cluster_centers_ is None or len(self.cluster_centers_) == 0:
            raise RuntimeError("请先调用 load() 加载聚类中心")
        return self.assign_soft(embeddings, top_k)


def generate_cluster_labels(
    llm_client,
    cluster_texts: dict[int, list[str]],
) -> dict[int, str]:
    """使用 LLM 为每个聚类生成描述性标签

    参数:
        llm_client: LLMClient 实例
        cluster_texts: {cluster_id: [代表性评论文本列表]}

    返回:
        {cluster_id: "标签名称"}
    """
    labels = {}
    for cid, texts in sorted(cluster_texts.items()):
        samples = texts[:10]  # 最多取 10 条代表性评论
        joined = "\n".join(f"- {t[:100]}" for t in samples)
        prompt = f"""你是一个酒店评论分析专家。以下是某组相关评论的样本，请为这组评论生成一个简洁的中文类别标签（2-4个字）。

评论样本：
{joined}

请只输出标签文本，不要有任何其他内容。标签示例：房间设施、餐饮质量、前台服务、交通便利"""
        try:
            label = llm_client.generate(prompt, temperature=0.3).strip()
            labels[int(cid)] = label
            print(f"  聚类 {cid}: {label}")
        except Exception as e:
            labels[int(cid)] = f"类别{cid}"
            print(f"  聚类 {cid}: 生成失败 ({e})，使用默认标签")

    return labels


def build_cluster_summaries(
    llm_client,
    cluster_comments: dict[int, list[str]],
    cluster_labels: dict[int, str],
) -> list[dict]:
    """为每个聚类生成摘要

    返回:
        [{"category": "标签名", "keywords": "关键词", "summary": "摘要", "comment_count": N}, ...]
    """
    summaries = []
    for cid in sorted(cluster_comments.keys()):
        comments = cluster_comments[cid]
        label = cluster_labels.get(cid, f"类别{cid}")
        # 取前 10 条评论的前 150 字作为关键词来源
        keywords = " ".join(c[:150] for c in comments[:10])

        joined_comments = "\n".join(f"- {c[:200]}" for c in comments[:15])
        prompt = f"""你是广州花园酒店的评论分析专家。请根据以下真实住客评论，写一段简洁的类别摘要。

类别：{label}
评论数：{len(comments)}

评论样本：
{joined_comments}

要求：
1. 概括该类别的主要正面观点和负面观点
2. 不超过 300 字
3. 语言客观、简洁"""
        try:
            summary = llm_client.generate(prompt, temperature=0.5).strip()
        except Exception:
            summary = f"关于{label}，共有{len(comments)}条相关评论。"

        summaries.append({
            "category": label,
            "keywords": keywords,
            "summary": summary,
            "comment_count": len(comments),
        })

    return summaries
