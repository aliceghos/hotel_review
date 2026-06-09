"""
多模态检索效果评估模块

评估多模态检索功能的效果，对比单模态 vs 多模态检索效果
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class RetrievalResult:
    """检索结果"""
    query_id: str
    retrieved_items: List[Dict]  # [{"item_id": str, "score": float, "modality": str}]
    retrieval_time: float = 0.0


@dataclass
class EvaluationMetrics:
    """评估指标"""
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0  # Mean Average Precision


@dataclass
class ContrastResult:
    """单模态vs多模态对比结果"""
    single_modality: EvaluationMetrics
    multimodal: EvaluationMetrics
    improvement: Dict[str, float]  # 各指标的提升百分比


@dataclass
class ResolutionImpactResult:
    """低分辨率影响分析结果"""
    original_recall: float
    super_resolution_recall: float
    improvement: float
    original_precision: float
    super_resolution_precision: float


class MultimodalEvaluator:
    """多模态检索效果评估器"""

    def __init__(self, k_values: List[int] = None):
        """
        初始化评估器

        Args:
            k_values: 评估时使用的k值列表，默认 [5, 10, 20]
        """
        self.k_values = k_values or [5, 10, 20]

    def evaluate_image_retrieval_recall(
        self,
        retrieval_results: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        评估图片检索召回率

        Args:
            retrieval_results: 检索结果列表，格式:
                [{"item_id": "img1", "score": 0.9}, ...]
            ground_truth: 正确答案列表，包含所有相关图片的ID
            k: 评估时考虑的top-k结果

        Returns:
            召回率 (Recall@K)
        """
        if not ground_truth:
            return 0.0

        if not retrieval_results:
            return 0.0

        # 取top-k结果
        top_k_results = retrieval_results[:k]
        retrieved_ids = set(item.get("item_id", "") for item in top_k_results)

        # 计算召回率
        relevant_set = set(ground_truth)
        retrieved_relevant = retrieved_ids & relevant_set

        recall = len(retrieved_relevant) / len(relevant_set) if relevant_set else 0.0

        return recall

    def evaluate_image_retrieval_precision(
        self,
        retrieval_results: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        评估图片检索精确率

        Args:
            retrieval_results: 检索结果列表
            ground_truth: 正确答案列表
            k: 评估时考虑的top-k结果

        Returns:
            精确率 (Precision@K)
        """
        if not retrieval_results:
            return 0.0

        top_k_results = retrieval_results[:k]
        retrieved_ids = set(item.get("item_id", "") for item in top_k_results)
        relevant_set = set(ground_truth)
        retrieved_relevant = retrieved_ids & relevant_set

        precision = len(retrieved_relevant) / k if k > 0 else 0.0

        return precision

    def evaluate_image_retrieval_f1(
        self,
        retrieval_results: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        评估图片检索F1分数

        Args:
            retrieval_results: 检索结果列表
            ground_truth: 正确答案列表
            k: 评估时考虑的top-k结果

        Returns:
            F1分数 (F1@K)
        """
        precision = self.evaluate_image_retrieval_precision(retrieval_results, ground_truth, k)
        recall = self.evaluate_image_retrieval_recall(retrieval_results, ground_truth, k)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def evaluate_multimodal_f1(
        self,
        text_results: List[Dict],
        image_results: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        评估多模态检索F1分数

        多模态检索综合了文本和图片检索结果

        Args:
            text_results: 文本检索结果
            image_results: 图片检索结果
            ground_truth: 正确答案列表
            k: 评估时考虑的top-k结果

        Returns:
            多模态F1分数
        """
        # 合并文本和图片结果，按分数排序
        combined_results = []

        for item in text_results:
            combined_results.append({
                "item_id": item.get("item_id", ""),
                "score": item.get("score", 0.0) * 0.5,  # 文本权重
                "modality": "text"
            })

        for item in image_results:
            combined_results.append({
                "item_id": item.get("item_id", ""),
                "score": item.get("score", 0.0) * 0.5,  # 图片权重
                "modality": "image"
            })

        # 按分数排序
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        # 取top-k去重
        seen_ids = set()
        deduped_results = []
        for item in combined_results:
            item_id = item["item_id"]
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped_results.append(item)

        return self.evaluate_image_retrieval_f1(deduped_results, ground_truth, k)

    def evaluate_cross_modal_relevance(
        self,
        query: str,
        retrieved_images: List[Dict]
    ) -> float:
        """
        评估跨模态相关性

        评估检索返回的图片与文本查询之间的相关性

        Args:
            query: 文本查询
            retrieved_images: 检索返回的图片列表

        Returns:
            跨模态相关性分数 (0-1)
        """
        if not retrieved_images:
            return 0.0

        total_relevance = 0.0

        for img_result in retrieved_images:
            # 图片与查询的相关性可以根据以下因素评估:
            # 1. 图片描述与查询的文本相似度
            # 2. 图片的语义标签与查询的匹配度
            # 3. 其他元数据

            description = img_result.get("description", "")
            semantic_tags = img_result.get("semantic_tags", [])

            # 简单的词汇重叠计算
            query_words = set(query.lower().split())
            desc_words = set(description.lower().split()) if description else set()

            # 描述匹配分数
            if desc_words:
                overlap = len(query_words & desc_words)
                desc_score = overlap / len(query_words) if query_words else 0
            else:
                desc_score = 0

            # 语义标签匹配分数
            tag_score = 0
            if semantic_tags:
                query_lower = query.lower()
                for tag in semantic_tags:
                    if tag.lower() in query_lower:
                        tag_score += 1
                tag_score = tag_score / len(semantic_tags)

            # 综合相关性分数
            relevance = 0.7 * desc_score + 0.3 * tag_score
            total_relevance += relevance

        # 平均相关性
        avg_relevance = total_relevance / len(retrieved_images) if retrieved_images else 0.0

        return avg_relevance

    def calculate_ndcg(
        self,
        retrieval_results: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        计算NDCG (Normalized Discounted Cumulative Gain)

        Args:
            retrieval_results: 检索结果列表
            ground_truth: 正确答案列表
            k: 评估时考虑的top-k结果

        Returns:
            NDCG@K分数
        """
        if not ground_truth or not retrieval_results:
            return 0.0

        relevant_set = set(ground_truth)

        # 计算DCG
        dcg = 0.0
        for i, item in enumerate(retrieval_results[:k]):
            item_id = item.get("item_id", "")
            if item_id in relevant_set:
                #  relevance = 1 if item in ground_truth else 0
                # 使用排名折扣
                dcg += 1.0 / np.log2(i + 2)  # i+2因为i从0开始

        # 计算IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)

        # 计算NDCG
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg

        return ndcg

    def calculate_map(
        self,
        retrieval_results: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        计算MAP (Mean Average Precision)

        Args:
            retrieval_results: 检索结果列表
            ground_truth: 正确答案列表
            k: 评估时考虑的top-k结果

        Returns:
            MAP@K分数
        """
        if not ground_truth or not retrieval_results:
            return 0.0

        relevant_set = set(ground_truth)

        precision_sum = 0.0
        relevant_count = 0

        for i, item in enumerate(retrieval_results[:k]):
            item_id = item.get("item_id", "")

            if item_id in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        if relevant_count == 0:
            return 0.0

        ap = precision_sum / len(relevant_set)

        return ap

    def compare_single_vs_multimodal(
        self,
        test_queries: List[Dict]
    ) -> Dict:
        """
        对比单模态(纯文本)和多模态检索效果

        Args:
            test_queries: 测试查询列表，格式:
                [{
                    "query_id": "q1",
                    "query": "酒店房间照片",
                    "relevant_images": ["img1", "img2"],
                    "relevant_texts": ["comment1", "comment2"],
                    "text_results": [{"item_id": "xxx", "score": 0.9}],
                    "image_results": [{"item_id": "xxx", "score": 0.8}],
                    "multimodal_results": [{"item_id": "xxx", "score": 0.85}]
                }]

        Returns:
            对比结果字典，包含各指标对比
        """
        results = {
            "summary": {},
            "per_query": [],
            "k_values": self.k_values
        }

        for k in self.k_values:
            single_recalls = []
            single_precisions = []
            single_f1s = []
            single_ndcgs = []

            multi_recalls = []
            multi_precisions = []
            multi_f1s = []
            multi_ndcgs = []

            for test_case in test_queries:
                query_id = test_case.get("query_id", "")
                relevant_images = test_case.get("relevant_images", [])
                text_results = test_case.get("text_results", [])
                image_results = test_case.get("image_results", [])
                multimodal_results = test_case.get("multimodal_results", [])

                # 单模态(文本)评估
                single_recall = self.evaluate_image_retrieval_recall(
                    text_results, relevant_images, k
                )
                single_precision = self.evaluate_image_retrieval_precision(
                    text_results, relevant_images, k
                )
                single_f1 = self.evaluate_image_retrieval_f1(
                    text_results, relevant_images, k
                )
                single_ndcg = self.calculate_ndcg(
                    text_results, relevant_images, k
                )

                single_recalls.append(single_recall)
                single_precisions.append(single_precision)
                single_f1s.append(single_f1)
                single_ndcgs.append(single_ndcg)

                # 多模态评估
                multi_recall = self.evaluate_image_retrieval_recall(
                    multimodal_results, relevant_images, k
                )
                multi_precision = self.evaluate_image_retrieval_precision(
                    multimodal_results, relevant_images, k
                )
                multi_f1 = self.evaluate_image_retrieval_f1(
                    multimodal_results, relevant_images, k
                )
                multi_ndcg = self.calculate_ndcg(
                    multimodal_results, relevant_images, k
                )

                multi_recalls.append(multi_recall)
                multi_precisions.append(multi_precision)
                multi_f1s.append(multi_f1)
                multi_ndcgs.append(multi_ndcg)

            # 计算平均值
            results["summary"][f"k_{k}"] = {
                "single_modality": {
                    "recall": np.mean(single_recalls) if single_recalls else 0.0,
                    "precision": np.mean(single_precisions) if single_precisions else 0.0,
                    "f1": np.mean(single_f1s) if single_f1s else 0.0,
                    "ndcg": np.mean(single_ndcgs) if single_ndcgs else 0.0
                },
                "multimodal": {
                    "recall": np.mean(multi_recalls) if multi_recalls else 0.0,
                    "precision": np.mean(multi_precisions) if multi_precisions else 0.0,
                    "f1": np.mean(multi_f1s) if multi_f1s else 0.0,
                    "ndcg": np.mean(multi_ndcgs) if multi_ndcgs else 0.0
                },
                "improvement": {
                    "recall_improvement": self._calculate_improvement(
                        np.mean(single_recalls), np.mean(multi_recalls)
                    ),
                    "precision_improvement": self._calculate_improvement(
                        np.mean(single_precisions), np.mean(multi_precisions)
                    ),
                    "f1_improvement": self._calculate_improvement(
                        np.mean(single_f1s), np.mean(multi_f1s)
                    )
                }
            }

        return results

    def _calculate_improvement(self, baseline: float, improved: float) -> float:
        """计算提升百分比"""
        if baseline == 0:
            return float("inf") if improved > 0 else 0.0
        return ((improved - baseline) / baseline) * 100

    def analyze_low_resolution_impact(
        self,
        retrieval_results: List[Dict]
    ) -> Dict:
        """
        分析低分辨率图片对检索的影响

        对比:
        - 超分辨率前的检索效果
        - 超分辨率后的检索效果

        Args:
            retrieval_results: 检索结果列表，格式:
                [{
                    "query_id": "q1",
                    "before_sr": {
                        "text_results": [...],
                        "image_results": [...]
                    },
                    "after_sr": {
                        "text_results": [...],
                        "image_results": [...]
                    },
                    "ground_truth": ["img1", "img2"]
                }]

        Returns:
            低分辨率影响分析结果
        """
        k = self.k_values[0] if self.k_values else 10

        results = {
            "summary": {},
            "per_query": []
        }

        before_recalls = []
        after_recalls = []
        before_precisions = []
        after_precisions = []

        for test_case in retrieval_results:
            query_id = test_case.get("query_id", "")
            ground_truth = test_case.get("ground_truth", [])

            before_sr = test_case.get("before_sr", {})
            after_sr = test_case.get("after_sr", {})

            # 超分辨率前
            before_text = before_sr.get("text_results", [])
            before_image = before_sr.get("image_results", [])
            before_combined = before_text + before_image
            before_combined.sort(key=lambda x: x.get("score", 0), reverse=True)

            before_recall = self.evaluate_image_retrieval_recall(
                before_combined, ground_truth, k
            )
            before_precision = self.evaluate_image_retrieval_precision(
                before_combined, ground_truth, k
            )

            # 超分辨率后
            after_text = after_sr.get("text_results", [])
            after_image = after_sr.get("image_results", [])
            after_combined = after_text + after_image
            after_combined.sort(key=lambda x: x.get("score", 0), reverse=True)

            after_recall = self.evaluate_image_retrieval_recall(
                after_combined, ground_truth, k
            )
            after_precision = self.evaluate_image_retrieval_precision(
                after_combined, ground_truth, k
            )

            before_recalls.append(before_recall)
            after_recalls.append(after_recall)
            before_precisions.append(before_precision)
            after_precisions.append(after_precision)

            results["per_query"].append({
                "query_id": query_id,
                "before_sr_recall": before_recall,
                "after_sr_recall": after_recall,
                "before_sr_precision": before_precision,
                "after_sr_precision": after_precision,
                "recall_improvement": after_recall - before_recall,
                "precision_improvement": after_precision - before_precision
            })

        # 计算总体改善
        avg_before_recall = np.mean(before_recalls) if before_recalls else 0.0
        avg_after_recall = np.mean(after_recalls) if after_recalls else 0.0
        avg_before_precision = np.mean(before_precisions) if before_precisions else 0.0
        avg_after_precision = np.mean(after_precisions) if after_precisions else 0.0

        results["summary"] = {
            "k": k,
            "before_super_resolution": {
                "avg_recall": avg_before_recall,
                "avg_precision": avg_before_precision
            },
            "after_super_resolution": {
                "avg_recall": avg_after_recall,
                "avg_precision": avg_after_precision
            },
            "improvement": {
                "recall_improvement": avg_after_recall - avg_before_recall,
                "recall_improvement_percent": self._calculate_improvement(
                    avg_before_recall, avg_after_recall
                ),
                "precision_improvement": avg_after_precision - avg_before_precision,
                "precision_improvement_percent": self._calculate_improvement(
                    avg_before_precision, avg_after_precision
                )
            }
        }

        return results

    def analyze_retrieval_path_contribution(
        self,
        retrieval_results: List[Dict]
    ) -> Dict:
        """
        分析各检索路径的贡献度

        Args:
            retrieval_results: 检索结果列表，格式:
                [{
                    "query_id": "q1",
                    "text_only_results": [...],  # 仅文本检索
                    "image_only_results": [...],  # 仅图片检索
                    "cross_modal_results": [...],  # 跨模态检索
                    "final_results": [...],  # 最终融合结果
                    "ground_truth": ["img1", "img2"]
                }]

        Returns:
            各检索路径贡献度分析
        """
        k = self.k_values[0] if self.k_values else 10

        text_contributions = []
        image_contributions = []
        cross_modal_contributions = []
        fusion_contributions = []

        for test_case in retrieval_results:
            ground_truth = set(test_case.get("ground_truth", []))

            # 各路径的命中情况
            text_only = test_case.get("text_only_results", [])[:k]
            image_only = test_case.get("image_only_results", [])[:k]
            cross_modal = test_case.get("cross_modal_results", [])[:k]
            final = test_case.get("final_results", [])[:k]

            text_hit = len(set(item.get("item_id", "") for item in text_only) & ground_truth)
            image_hit = len(set(item.get("item_id", "") for item in image_only) & ground_truth)
            cross_modal_hit = len(set(item.get("item_id", "") for item in cross_modal) & ground_truth)
            final_hit = len(set(item.get("item_id", "") for item in final) & ground_truth)

            # 计算贡献度 (该路径独有贡献 / 总贡献)
            # 简化版本：计算各路径的召回贡献
            total_hit = len(ground_truth)
            if total_hit > 0:
                text_contributions.append(text_hit / total_hit)
                image_contributions.append(image_hit / total_hit)
                cross_modal_contributions.append(cross_modal_hit / total_hit)
                fusion_contributions.append(final_hit / total_hit)

        results = {
            "summary": {
                "text_retrieval_contribution": np.mean(text_contributions) if text_contributions else 0.0,
                "image_retrieval_contribution": np.mean(image_contributions) if image_contributions else 0.0,
                "cross_modal_contribution": np.mean(cross_modal_contributions) if cross_modal_contributions else 0.0,
                "fusion_contribution": np.mean(fusion_contributions) if fusion_contributions else 0.0
            },
            "contribution_percentage": {
                "text_retrieval": 0.0,
                "image_retrieval": 0.0,
                "cross_modal_retrieval": 0.0
            }
        }

        # 转换为百分比
        total_contribution = (
            results["summary"]["text_retrieval_contribution"] +
            results["summary"]["image_retrieval_contribution"] +
            results["summary"]["cross_modal_contribution"]
        )

        if total_contribution > 0:
            results["contribution_percentage"]["text_retrieval"] = (
                results["summary"]["text_retrieval_contribution"] / total_contribution * 100
            )
            results["contribution_percentage"]["image_retrieval"] = (
                results["summary"]["image_retrieval_contribution"] / total_contribution * 100
            )
            results["contribution_percentage"]["cross_modal_retrieval"] = (
                results["summary"]["cross_modal_contribution"] / total_contribution * 100
            )

        return results

    def generate_report(
        self,
        results: Dict,
        report_type: str = "full"
    ) -> str:
        """
        生成评估报告

        Args:
            results: 评估结果字典，包含以下键:
                - comparison_results: 单模态vs多模态对比结果
                - image_recall_results: 图片检索召回率结果
                - cross_modal_relevance: 跨模态相关性结果
                - resolution_impact: 低分辨率影响分析结果
                - contribution_analysis: 贡献度分析结果
            report_type: 报告类型 ("full", "summary", "brief")

        Returns:
            格式化后的评估报告字符串
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if report_type == "brief":
            return self._generate_brief_report(results, timestamp)
        elif report_type == "summary":
            return self._generate_summary_report(results, timestamp)
        else:
            return self._generate_full_report(results, timestamp)

    def _generate_full_report(self, results: Dict, timestamp: str) -> str:
        """生成完整报告"""
        report = f"""# 多模态检索效果评估报告

**生成时间**: {timestamp}

---

## 1. 评估概述

### 1.1 评估目的
评估多模态检索功能的效果，对比单模态(纯文本)检索与多模态检索的性能差异，
分析图片检索对整体系统的贡献，以及低分辨率图片对检索效果的影响。

### 1.2 测试数据集
"""

        # 添加测试数据集信息
        if "test_info" in results:
            test_info = results["test_info"]
            report += f"""
- 测试查询数量: {test_info.get("query_count", "N/A")}
- 平均每查询相关图片数: {test_info.get("avg_relevant_images", "N/A")}
- 平均每查询相关文本数: {test_info.get("avg_relevant_texts", "N/A")}
"""
        else:
            report += """
- 测试查询数量: N/A
- 平均每查询相关图片数: N/A
"""

        report += """
### 1.3 评估指标
- **Recall@K**: 召回率，衡量检索系统找回相关结果的能力
- **Precision@K**: 精确率，衡量检索结果中相关结果的比例
- **F1@K**: 精确率和召回率的调和平均
- **NDCG@K**: 归一化折损累计增益，考虑结果排序质量
- **跨模态相关性**: 文本查询与检索图片之间的语义相关性

---

## 2. 单模态 vs 多模态检索对比

"""

        # 添加对比表格
        comparison = results.get("comparison_results", {})
        summary = comparison.get("summary", {})

        if summary:
            report += "| 指标 | 单模态(文本) | 多模态 | 提升 |\\n"
            report += "|------|-------------|--------|------|\\n"

            for k_key, k_data in summary.items():
                k = k_key.replace("k_", "")
                single = k_data.get("single_modality", {})
                multi = k_data.get("multimodal", {})
                improvement = k_data.get("improvement", {})

                report += f"| Recall@{k} | {single.get('recall', 0):.4f} | {multi.get('recall', 0):.4f} | {improvement.get('recall_improvement', 0):.2f}% |\\n"
                report += f"| Precision@{k} | {single.get('precision', 0):.4f} | {multi.get('precision', 0):.4f} | {improvement.get('precision_improvement', 0):.2f}% |\\n"
                report += f"| F1@{k} | {single.get('f1', 0):.4f} | {multi.get('f1', 0):.4f} | {improvement.get('f1_improvement', 0):.2f}% |\\n"

        report += """
---

## 3. 图片检索专项评估

"""

        image_recall = results.get("image_recall_results", {})
        if image_recall:
            report += f"""
| 指标 | 数值 |
|------|------|
| 图片检索召回率 | {image_recall.get('avg_recall', 0):.4f} |
| 图片检索精确率 | {image_recall.get('avg_precision', 0):.4f} |
| 图片检索F1 | {image_recall.get('avg_f1', 0):.4f} |
"""
        else:
            report += "| 指标 | 数值 |\\n|------|------|\\n"

        cross_modal = results.get("cross_modal_relevance", {})
        if cross_modal:
            report += f"""
| 跨模态相关性 | {cross_modal.get('avg_relevance', 0):.4f} |
"""
        else:
            report += "| 跨模态相关性 | N/A |\\n"

        report += """
---

## 4. 低分辨率图片影响分析

"""

        resolution_impact = results.get("resolution_impact", {})
        impact_summary = resolution_impact.get("summary", {})

        if impact_summary:
            k = impact_summary.get("k", 10)
            before = impact_summary.get("before_super_resolution", {})
            after = impact_summary.get("after_super_resolution", {})
            improvement = impact_summary.get("improvement", {})

            report += f"""
| 处理方式 | Recall@{k} | Precision@{k} |
|----------|------------|---------------|
| 原图(低分辨率) | {before.get('avg_recall', 0):.4f} | {before.get('avg_precision', 0):.4f} |
| 超分辨率后 | {after.get('avg_recall', 0):.4f} | {after.get('avg_precision', 0):.4f} |
| 改善 | +{improvement.get('recall_improvement', 0):.4f} ({improvement.get('recall_improvement_percent', 0):.2f}%) | +{improvement.get('precision_improvement', 0):.4f} |

"""
        else:
            report += "| 处理方式 | 检索效果 |\\n|----------|----------|\\n"

        report += """
---

## 5. 各检索路径贡献度

"""

        contribution = results.get("contribution_analysis", {})
        contribution_pct = contribution.get("contribution_percentage", {})

        if contribution_pct:
            report += f"""
| 检索路径 | 贡献度 |
|----------|--------|
| 文本检索 | {contribution_pct.get('text_retrieval', 0):.1f}% |
| 图片检索 | {contribution_pct.get('image_retrieval', 0):.1f}% |
| 跨模态检索 | {contribution_pct.get('cross_modal_retrieval', 0):.1f}% |

"""
        else:
            report += "| 检索路径 | 贡献度 |\\n|----------|--------|\\n"

        report += """
---

## 6. 结论与建议

### 6.1 主要发现
"""

        # 自动生成结论
        if summary:
            for k_key, k_data in list(summary.items())[:1]:
                k = k_key.replace("k_", "")
                improvement = k_data.get("improvement", {})
                f1_improvement = improvement.get("f1_improvement", 0)

                if f1_improvement > 10:
                    report += f"""
1. **多模态检索效果显著提升**: 相比单模态(文本)检索，多模态检索的F1@{k}提升了{f1_improvement:.1f}%，
   表明结合图片信息能有效改善检索效果。
"""
                elif f1_improvement > 0:
                    report += f"""
1. **多模态检索有一定改善**: 多模态检索相比单模态检索有{f1_improvement:.1f}%的F1@{k}提升，
   图片信息对检索有一定帮助。
"""
                else:
                    report += """
1. **多模态检索效果持平或下降**: 需要检查图片质量和相关性计算方式。
"""

        if impact_summary:
            improvement = impact_summary.get("improvement", {})
            recall_imp = improvement.get('recall_improvement_percent', 0)
            if recall_imp > 5:
                report += f"""
2. **超分辨率效果明显**: 超分辨率处理后，图片检索召回率提升了{recall_imp:.1f}%，
   低分辨率图片是影响检索效果的重要因素。
"""
            elif recall_imp > 0:
                report += f"""
2. **超分辨率有一定帮助**: 超分辨率处理带来了{recall_imp:.1f}%的召回率提升。
"""

        if contribution_pct:
            img_contrib = contribution_pct.get('image_retrieval', 0)
            if img_contrib > 30:
                report += f"""
3. **图片检索贡献显著**: 图片检索路径贡献了{img_contrib:.1f}%的相关结果，
   建议进一步优化图片向量化模型。
"""

        report += """
### 6.2 优化建议
"""

        if contribution_pct:
            img_contrib = contribution_pct.get('image_retrieval', 0)
            if img_contrib < 20:
                report += """
1. **提升图片检索权重**: 当前图片检索贡献较低，建议:
   - 优化图片特征提取模型
   - 增加训练数据中的图片多样性
   - 调整图文融合策略
"""

        if impact_summary:
            improvement = impact_summary.get("improvement", {})
            recall_imp = improvement.get('recall_improvement_percent', 0)
            if recall_imp > 3:
                report += """
2. **部署超分辨率模型**: 超分辨率处理能有效提升检索效果，建议:
   - 在检索流水线中集成超分辨率模块
   - 针对低分辨率图片启用超分辨率
   - 评估超分辨率的计算开销与效果收益
"""

        report += """
---

*报告生成时间: {timestamp}*

""".format(timestamp=timestamp)

        return report

    def _generate_summary_report(self, results: Dict, timestamp: str) -> str:
        """生成摘要报告"""
        report = f"""# 多模态检索效果评估报告 (摘要)

**生成时间**: {timestamp}

## 核心指标

"""

        comparison = results.get("comparison_results", {})
        summary = comparison.get("summary", {})

        if summary:
            k_key = list(summary.keys())[0] if summary else "k_10"
            k_data = summary.get(k_key, {})
            k = k_key.replace("k_", "")
            single = k_data.get("single_modality", {})
            multi = k_data.get("multimodal", {})
            improvement = k_data.get("improvement", {})

            report += f"| 指标 | 单模态 | 多模态 | 提升 |\\n"
            report += f"|------|--------|-------|------|\\n"
            report += f"| F1@{k} | {single.get('f1', 0):.3f} | {multi.get('f1', 0):.3f} | {improvement.get('f1_improvement', 0):.1f}% |\\n"
            report += f"| Recall@{k} | {single.get('recall', 0):.3f} | {multi.get('recall', 0):.3f} | {improvement.get('recall_improvement', 0):.1f}% |\\n"

        contribution = results.get("contribution_analysis", {})
        contribution_pct = contribution.get("contribution_percentage", {})
        if contribution_pct:
            report += f"""
## 检索路径贡献
- 文本: {contribution_pct.get('text_retrieval', 0):.1f}%
- 图片: {contribution_pct.get('image_retrieval', 0):.1f}%
- 跨模态: {contribution_pct.get('cross_modal_retrieval', 0):.1f}%

"""

        return report

    def _generate_brief_report(self, results: Dict, timestamp: str) -> str:
        """生成简短报告"""
        comparison = results.get("comparison_results", {})
        summary = comparison.get("summary", {})

        brief = f"[{timestamp}] "

        if summary:
            k_key = list(summary.keys())[0] if summary else "k_10"
            k_data = summary.get(k_key, {})
            improvement = k_data.get("improvement", {})

            f1_imp = improvement.get('f1_improvement', 0)
            brief += f"多模态F1提升: {f1_imp:+.1f}%"

        return brief

    def save_report(self, report: str, output_path: str):
        """
        保存报告到文件

        Args:
            report: 报告内容
            output_path: 输出文件路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    def run_full_evaluation(
        self,
        test_queries: List[Dict],
        low_res_test_cases: List[Dict] = None,
        contribution_test_cases: List[Dict] = None
    ) -> Dict:
        """
        运行完整评估流程

        Args:
            test_queries: 测试查询列表，用于单模态vs多模态对比
            low_res_test_cases: 低分辨率测试用例
            contribution_test_cases: 贡献度分析测试用例

        Returns:
            完整评估结果
        """
        results = {
            "test_info": {
                "query_count": len(test_queries),
                "avg_relevant_images": np.mean([
                    len(q.get("relevant_images", [])) for q in test_queries
                ]) if test_queries else 0,
                "avg_relevant_texts": np.mean([
                    len(q.get("relevant_texts", [])) for q in test_queries
                ]) if test_queries else 0
            }
        }

        # 单模态 vs 多模态对比
        if test_queries:
            results["comparison_results"] = self.compare_single_vs_multimodal(test_queries)

            # 计算图片检索专项指标
            image_recalls = []
            image_precisions = []
            image_f1s = []
            cross_modal_relevances = []

            for test_case in test_queries:
                relevant_images = test_case.get("relevant_images", [])
                multimodal_results = test_case.get("multimodal_results", [])
                query = test_case.get("query", "")

                if relevant_images and multimodal_results:
                    recall = self.evaluate_image_retrieval_recall(
                        multimodal_results, relevant_images
                    )
                    precision = self.evaluate_image_retrieval_precision(
                        multimodal_results, relevant_images
                    )
                    f1 = self.evaluate_image_retrieval_f1(
                        multimodal_results, relevant_images
                    )

                    image_recalls.append(recall)
                    image_precisions.append(precision)
                    image_f1s.append(f1)

                # 跨模态相关性
                if query and multimodal_results:
                    relevance = self.evaluate_cross_modal_relevance(query, multimodal_results)
                    cross_modal_relevances.append(relevance)

            results["image_recall_results"] = {
                "avg_recall": np.mean(image_recalls) if image_recalls else 0,
                "avg_precision": np.mean(image_precisions) if image_precisions else 0,
                "avg_f1": np.mean(image_f1s) if image_f1s else 0
            }

            results["cross_modal_relevance"] = {
                "avg_relevance": np.mean(cross_modal_relevances) if cross_modal_relevances else 0
            }

        # 低分辨率影响分析
        if low_res_test_cases:
            results["resolution_impact"] = self.analyze_low_resolution_impact(low_res_test_cases)

        # 贡献度分析
        if contribution_test_cases:
            results["contribution_analysis"] = self.analyze_retrieval_path_contribution(
                contribution_test_cases
            )

        return results


# 示例测试数据生成函数
def generate_synthetic_test_data(n_queries: int = 20) -> List[Dict]:
    """
    生成合成测试数据用于演示评估

    Args:
        n_queries: 生成的测试查询数量

    Returns:
        测试数据列表
    """
    np.random.seed(42)

    test_queries = []

    query_templates = [
        "酒店房间照片",
        "餐厅环境",
        "健身房设施",
        "游泳池图片",
        "房间夜景",
        "早餐食品",
        "酒店外观",
        "浴室设施",
        "会议室照片",
        "客房服务"
    ]

    for i in range(n_queries):
        query = query_templates[i % len(query_templates)]
        n_relevant = np.random.randint(2, 6)

        # 生成相关图片ID
        relevant_images = [f"img_{i}_{j}" for j in range(n_relevant)]

        # 生成单模态(文本)检索结果
        text_results = []
        for j in range(10):
            img_id = f"img_{i}_{j % n_relevant}" if j < n_relevant * 2 else f"other_{i}_{j}"
            text_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.5, 0.95)
            })
        text_results.sort(key=lambda x: x["score"], reverse=True)

        # 生成图片检索结果
        image_results = []
        for j in range(10):
            img_id = f"img_{i}_{j % n_relevant}" if j < n_relevant else f"other_{i}_{j}"
            image_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.4, 0.9)
            })
        image_results.sort(key=lambda x: x["score"], reverse=True)

        # 生成多模态融合结果
        multimodal_results = []
        for j in range(10):
            img_id = f"img_{i}_{j % n_relevant}" if j < int(n_relevant * 1.2) else f"other_{i}_{j}"
            multimodal_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.55, 0.92)
            })
        multimodal_results.sort(key=lambda x: x["score"], reverse=True)

        test_queries.append({
            "query_id": f"q_{i}",
            "query": query,
            "relevant_images": relevant_images,
            "relevant_texts": [f"text_{i}_{j}" for j in range(n_relevant)],
            "text_results": text_results,
            "image_results": image_results,
            "multimodal_results": multimodal_results
        })

    return test_queries


def generate_low_resolution_test_data(n_cases: int = 15) -> List[Dict]:
    """生成低分辨率影响测试数据"""
    np.random.seed(43)

    test_cases = []

    for i in range(n_cases):
        n_relevant = np.random.randint(2, 5)
        relevant_images = [f"img_{i}_{j}" for j in range(n_relevant)]

        # 超分辨率前结果
        before_text = []
        before_image = []
        for j in range(10):
            img_id = f"img_{i}_{j % n_relevant}" if j < n_relevant else f"other_{i}_{j}"
            before_text.append({
                "item_id": img_id,
                "score": np.random.uniform(0.4, 0.85)
            })
            before_image.append({
                "item_id": img_id,
                "score": np.random.uniform(0.35, 0.8)
            })

        before_text.sort(key=lambda x: x["score"], reverse=True)
        before_image.sort(key=lambda x: x["score"], reverse=True)

        # 超分辨率后结果 (假设有改善)
        after_text = []
        after_image = []
        for j in range(10):
            img_id = f"img_{i}_{j % n_relevant}" if j < int(n_relevant * 1.2) else f"other_{i}_{j}"
            after_text.append({
                "item_id": img_id,
                "score": np.random.uniform(0.5, 0.9)
            })
            after_image.append({
                "item_id": img_id,
                "score": np.random.uniform(0.45, 0.88)
            })

        after_text.sort(key=lambda x: x["score"], reverse=True)
        after_image.sort(key=lambda x: x["score"], reverse=True)

        test_cases.append({
            "query_id": f"q_lr_{i}",
            "before_sr": {
                "text_results": before_text,
                "image_results": before_image
            },
            "after_sr": {
                "text_results": after_text,
                "image_results": after_image
            },
            "ground_truth": relevant_images
        })

    return test_cases


def generate_contribution_test_data(n_cases: int = 15) -> List[Dict]:
    """生成贡献度分析测试数据"""
    np.random.seed(44)

    test_cases = []

    for i in range(n_cases):
        n_relevant = np.random.randint(2, 5)
        relevant_images = set([f"img_{i}_{j}" for j in range(n_relevant)])

        # 文本检索独有命中的图片
        text_hit = set()
        for j in range(3):
            if np.random.random() > 0.5:
                text_hit.add(f"img_{i}_{j}")

        # 图片检索独有命中的图片
        image_hit = set()
        for j in range(2, 5):
            if np.random.random() > 0.4:
                image_hit.add(f"img_{i}_{j}")

        # 跨模态命中的图片
        cross_modal_hit = set()
        for j in range(1, 4):
            if np.random.random() > 0.3:
                cross_modal_hit.add(f"img_{i}_{j}")

        # 最终融合结果 (合并所有)
        final_hit = relevant_images

        # 生成结果列表
        text_only_results = []
        for img_id in text_hit:
            text_only_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.7, 0.95)
            })
        for j in range(10 - len(text_only_results)):
            text_only_results.append({
                "item_id": f"other_{i}_{j}",
                "score": np.random.uniform(0.3, 0.6)
            })
        text_only_results.sort(key=lambda x: x["score"], reverse=True)

        image_only_results = []
        for img_id in image_hit:
            image_only_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.65, 0.9)
            })
        for j in range(10 - len(image_only_results)):
            image_only_results.append({
                "item_id": f"other_{i}_{j}",
                "score": np.random.uniform(0.3, 0.55)
            })
        image_only_results.sort(key=lambda x: x["score"], reverse=True)

        cross_modal_results = []
        for img_id in cross_modal_hit:
            cross_modal_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.6, 0.88)
            })
        for j in range(10 - len(cross_modal_results)):
            cross_modal_results.append({
                "item_id": f"other_{i}_{j}",
                "score": np.random.uniform(0.35, 0.58)
            })
        cross_modal_results.sort(key=lambda x: x["score"], reverse=True)

        final_results = []
        for img_id in final_hit:
            final_results.append({
                "item_id": img_id,
                "score": np.random.uniform(0.75, 0.95)
            })
        for j in range(10 - len(final_results)):
            final_results.append({
                "item_id": f"other_{i}_{j}",
                "score": np.random.uniform(0.3, 0.5)
            })
        final_results.sort(key=lambda x: x["score"], reverse=True)

        test_cases.append({
            "query_id": f"q_contrib_{i}",
            "text_only_results": text_only_results,
            "image_only_results": image_only_results,
            "cross_modal_results": cross_modal_results,
            "final_results": final_results,
            "ground_truth": list(relevant_images)
        })

    return test_cases


if __name__ == "__main__":
    print("多模态检索效果评估模块")
    print("=" * 50)

    # 创建评估器
    evaluator = MultimodalEvaluator(k_values=[5, 10])

    # 生成测试数据
    print("\n1. 生成测试数据...")
    test_queries = generate_synthetic_test_data(n_queries=20)
    low_res_cases = generate_low_resolution_test_data(n_cases=15)
    contribution_cases = generate_contribution_test_data(n_cases=15)
    print(f"   - 测试查询: {len(test_queries)} 条")
    print(f"   - 低分辨率测试: {len(low_res_cases)} 条")
    print(f"   - 贡献度测试: {len(contribution_cases)} 条")

    # 运行完整评估
    print("\n2. 运行评估...")
    results = evaluator.run_full_evaluation(
        test_queries=test_queries,
        low_res_test_cases=low_res_cases,
        contribution_test_cases=contribution_cases
    )

    # 生成报告
    print("\n3. 生成评估报告...")
    report = evaluator.generate_report(results, report_type="full")
    print(report)

    # 保存报告
    output_path = "multimodal_evaluation_report.md"
    evaluator.save_report(report, output_path)
    print(f"\n报告已保存到: {output_path}")
