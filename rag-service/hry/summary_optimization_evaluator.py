"""
摘要优化效果评估器

评估多粒度摘要优化后的检索效果，对比单粒度 vs 多粒度检索。
"""

import json
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


# ============== 测试查询集 ==============

TEST_QUERIES = {
    "general": [
        "酒店整体怎么样？",
        "这家酒店值得入住吗？",
        "综合体验如何？"
    ],
    "aspect": [
        "前台服务态度好吗？",
        "房间安静吗？",
        "早餐丰富吗？"
    ],
    "specific": [
        "红棉大床房的床是什么品牌？",
        "泳池几点开放到几点？",
        "行政酒廊提供什么服务？"
    ]
}


# ============== 数据结构 ==============

@dataclass
class EvaluationMetrics:
    """评估指标"""
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass
class GranularityContribution:
    """各粒度贡献度"""
    category_contribution: float
    aspect_contribution: float
    comment_contribution: float


# ============== 评估器实现 ==============

class SummaryOptimizationEvaluator:
    """摘要优化效果评估器"""
    
    def __init__(self, indexer=None, adaptive_retriever=None):
        """
        初始化评估器
        
        Args:
            indexer: 多粒度索引器（用于单粒度对比）
            adaptive_retriever: 自适应检索器（用于多粒度对比）
        """
        self.indexer = indexer
        self.adaptive_retriever = adaptive_retriever
        self.evaluation_history: List[Dict] = []
    
    def _get_result_id(self, result) -> str:
        """获取结果ID，兼容dict和object"""
        if isinstance(result, dict):
            return result.get("item_id", result.get("id", ""))
        return getattr(result, "item_id", getattr(result, "id", ""))
    
    def _get_result_type(self, result) -> str:
        """获取结果类型，兼容dict和object"""
        if isinstance(result, dict):
            return result.get("item_type", result.get("source_level", ""))
        return getattr(result, "item_type", getattr(result, "source_level", ""))
    
    def _get_result_score(self, result) -> float:
        """获取结果分数，兼容dict和object"""
        if isinstance(result, dict):
            return result.get("score", result.get("relevance_score", 0.5))
        return getattr(result, "score", 0.5)
    
    def evaluate_recall(self, retrieval_results: list, ground_truth: list, k: int) -> float:
        """
        计算Recall@K
        
        Args:
            retrieval_results: 检索结果列表，每项包含 item_id
            ground_truth: 正确答案ID列表
            k: 截断位置
        
        Returns:
            Recall@K 值
        """
        if not ground_truth:
            return 0.0
        
        # 获取Top-K结果
        top_k_results = retrieval_results[:k] if k > 0 else retrieval_results
        
        # 提取结果ID
        retrieved_ids = set()
        for result in top_k_results:
            item_id = self._get_result_id(result)
            retrieved_ids.add(item_id)
        
        # 计算命中数
        hits = len(retrieved_ids & set(ground_truth))
        
        return hits / len(ground_truth) if ground_truth else 0.0
    
    def evaluate_mrr(self, retrieval_results: list, ground_truth: list) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        
        MRR = 1/|Q| * sum(1/rank_i)
        其中rank_i是第i个查询的第一个正确答案的排名
        
        Args:
            retrieval_results: 检索结果列表
            ground_truth: 正确答案ID列表
        
        Returns:
            MRR 值
        """
        if not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        
        # 查找第一个正确答案的排名
        for i, result in enumerate(retrieval_results, 1):
            item_id = self._get_result_id(result)
            if item_id in ground_truth_set:
                return 1.0 / i
        
        return 0.0
    
    def evaluate_ndcg(self, retrieval_results: list, ground_truth: list, k: int) -> float:
        """
        计算NDCG@K (Normalized Discounted Cumulative Gain)
        
        DCG@K = sum(i=1 to K, rel_i / log2(i+1))
        NDCG@K = DCG@K / IDCG@K
        
        Args:
            retrieval_results: 检索结果列表
            ground_truth: 正确答案ID列表
            k: 截断位置
        
        Returns:
            NDCG@K 值
        """
        if not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        
        # 计算DCG
        dcg = 0.0
        top_k_results = retrieval_results[:k] if k > 0 else retrieval_results
        
        for i, result in enumerate(top_k_results, 1):
            item_id = self._get_result_id(result)
            # 相关性为1如果在ground truth中，否则为0
            relevance = 1.0 if item_id in ground_truth_set else 0.0
            dcg += relevance / math.log2(i + 1)
        
        # 计算IDCG（理想DCG）
        idcg = 0.0
        num_relevant = min(len(ground_truth), k)
        for i in range(1, num_relevant + 1):
            idcg += 1.0 / math.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_single_granularity(
        self,
        query: str,
        ground_truth: list,
        granularity: str = "comment",
        k: int = 10
    ) -> EvaluationMetrics:
        """
        评估单粒度检索效果
        
        Args:
            query: 查询文本
            ground_truth: 正确答案列表
            granularity: 粒度类型 ("category", "aspect", "comment")
            k: 评估截断位置
        
        Returns:
            评估指标
        """
        if self.indexer is None:
            raise RuntimeError("索引器未初始化")
        
        # 执行单粒度检索
        results = self.indexer.retrieve_with_fusion(
            query=query,
            topk=k,
            weights={granularity: 1.0, "category": 0.0, "aspect": 0.0, "comment": 0.0}
        )
        
        # 转换为统一格式
        formatted_results = [
            {
                "item_id": r.item_id,
                "text": r.text,
                "score": r.score,
                "item_type": r.item_type
            }
            for r in results
        ]
        
        # 计算指标
        recall = self.evaluate_recall(formatted_results, ground_truth, k)
        mrr = self.evaluate_mrr(formatted_results, ground_truth)
        ndcg = self.evaluate_ndcg(formatted_results, ground_truth, k)
        
        return EvaluationMetrics(
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg
        )
    
    def evaluate_multi_granularity(
        self,
        query: str,
        ground_truth: list,
        k: int = 10
    ) -> EvaluationMetrics:
        """
        评估多粒度检索效果
        
        Args:
            query: 查询文本
            ground_truth: 正确答案列表
            k: 评估截断位置
        
        Returns:
            评估指标
        """
        if self.adaptive_retriever is None and self.indexer is None:
            raise RuntimeError("检索器未初始化")
        
        if self.adaptive_retriever:
            # 使用自适应检索器
            results = self.adaptive_retriever.retrieve(query, topk=k)
        else:
            # 使用默认融合
            results = self.indexer.retrieve_with_fusion(query=query, topk=k)
        
        # 计算指标
        recall = self.evaluate_recall(results, ground_truth, k)
        mrr = self.evaluate_mrr(results, ground_truth)
        ndcg = self.evaluate_ndcg(results, ground_truth, k)
        
        return EvaluationMetrics(
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg
        )
    
    def compare_single_vs_multi(
        self,
        queries: List[str],
        ground_truth: Dict[str, list]
    ) -> dict:
        """
        对比单粒度和多粒度检索效果
        
        Args:
            queries: 查询列表
            ground_truth: 查询对应的正确答案字典 {query: [item_ids]}
        
        Returns:
            对比结果字典
        """
        k = 10
        metrics_history = {
            "single_granularity": {"category": [], "aspect": [], "comment": []},
            "multi_granularity": []
        }
        
        for query in queries:
            gt = ground_truth.get(query, [])
            
            # 单粒度评估
            for granularity in ["category", "aspect", "comment"]:
                try:
                    metrics = self.evaluate_single_granularity(
                        query, gt, granularity, k
                    )
                    metrics_history["single_granularity"][granularity].append(asdict(metrics))
                except Exception as e:
                    print(f"  警告: {granularity}粒度评估失败 ({query}): {e}")
            
            # 多粒度评估
            try:
                metrics = self.evaluate_multi_granularity(query, gt, k)
                metrics_history["multi_granularity"].append(asdict(metrics))
            except Exception as e:
                print(f"  警告: 多粒度评估失败 ({query}): {e}")
        
        # 汇总计算
        def avg_metrics(metrics_list: list) -> dict:
            if not metrics_list:
                return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
            return {
                "recall_at_k": sum(m["recall_at_k"] for m in metrics_list) / len(metrics_list),
                "mrr": sum(m["mrr"] for m in metrics_list) / len(metrics_list),
                "ndcg_at_k": sum(m["ndcg_at_k"] for m in metrics_list) / len(metrics_list)
            }
        
        # 计算单粒度平均（取最佳粒度）
        best_single = {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
        for granularity in ["category", "aspect", "comment"]:
            gran_avg = avg_metrics(metrics_history["single_granularity"][granularity])
            if gran_avg["recall_at_k"] > best_single["recall_at_k"]:
                best_single = gran_avg
        
        # 多粒度平均
        multi_avg = avg_metrics(metrics_history["multi_granularity"])
        
        # 计算提升
        def calc_improvement(baseline: dict, improved: dict) -> dict:
            return {
                "recall@10": f"{(improved['recall_at_k'] - baseline['recall_at_k']) * 100:+.1f}%",
                "mrr": f"{(improved['mrr'] - baseline['mrr']) * 100:+.1f}%",
                "ndcg@10": f"{(improved['ndcg_at_k'] - baseline['ndcg_at_k']) * 100:+.1f}%"
            }
        
        improvement = calc_improvement(best_single, multi_avg)
        
        return {
            "single_granularity": {
                "recall@10": round(best_single["recall_at_k"], 4),
                "mrr": round(best_single["mrr"], 4),
                "ndcg@10": round(best_single["ndcg_at_k"], 4)
            },
            "multi_granularity": {
                "recall@10": round(multi_avg["recall_at_k"], 4),
                "mrr": round(multi_avg["mrr"], 4),
                "ndcg@10": round(multi_avg["ndcg_at_k"], 4)
            },
            "improvement": improvement,
            "detailed_metrics": metrics_history
        }
    
    def analyze_granularity_contribution(self, retrieval_results: list) -> dict:
        """
        分析各粒度摘要的贡献度
        
        贡献度定义：在Top-K结果中，该粒度结果被用户点击/采纳的比例
        
        Args:
            retrieval_results: 多粒度检索结果列表
        
        Returns:
            各粒度贡献度字典
        """
        if not retrieval_results:
            return {
                "category_contribution": 0.0,
                "aspect_contribution": 0.0,
                "comment_contribution": 0.0
            }
        
        # 统计各粒度结果数量
        granularity_counts = {"category": 0, "aspect": 0, "comment": 0}
        total = len(retrieval_results)
        
        for result in retrieval_results:
            item_type = self._get_result_type(result)
            if item_type in granularity_counts:
                granularity_counts[item_type] += 1
        
        # 计算贡献度（比例）
        contribution = {
            "category_contribution": granularity_counts["category"] / total if total > 0 else 0.0,
            "aspect_contribution": granularity_counts["aspect"] / total if total > 0 else 0.0,
            "comment_contribution": granularity_counts["comment"] / total if total > 0 else 0.0
        }
        
        return contribution
    
    def analyze_query_type_effectiveness(
        self,
        queries_by_type: dict,
        ground_truth: dict
    ) -> dict:
        """
        分析不同查询类型的效果
        
        Args:
            queries_by_type: 按类型分组的查询 {"general": [...], "aspect": [...], "specific": [...]}
            ground_truth: 查询对应的正确答案
        
        Returns:
            各查询类型的效果分析
        """
        results = {}
        
        for query_type, queries in queries_by_type.items():
            type_results = self.compare_single_vs_multi(queries, ground_truth)
            results[query_type] = type_results
        
        return results
    
    def generate_report(self, evaluation_results: dict) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果字典
        
        Returns:
            Markdown格式的报告
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取数据
        single = evaluation_results.get("single_granularity", {})
        multi = evaluation_results.get("multi_granularity", {})
        improvement = evaluation_results.get("improvement", {})
        
        # 查询类型效果
        query_type_results = evaluation_results.get("query_type_effectiveness", {})
        
        # 各粒度贡献
        contribution = evaluation_results.get("granularity_contribution", {})
        
        report = f"""# 摘要优化效果评估报告

**生成时间**: {timestamp}

---

## 1. 评估概述

### 评估方法
- **单粒度检索**: 分别使用类别级、观点级、评论级索引进行独立检索，取最佳结果
- **多粒度检索**: 融合类别级、观点级、评论级检索结果
- **融合权重**: category=0.2, aspect=0.3, comment=0.5

### 测试数据集
- 泛化查询 (general): {len(TEST_QUERIES.get("general", []))} 条
- 观点查询 (aspect): {len(TEST_QUERIES.get("aspect", []))} 条
- 具体查询 (specific): {len(TEST_QUERIES.get("specific", []))} 条
- 共 {sum(len(v) for v in TEST_QUERIES.values())} 条测试查询

### 评估指标
- **Recall@K**: 召回率，衡量检索系统找到相关结果的能力
- **MRR**: 平均倒数排名，衡量第一个相关结果的位置
- **NDCG@K**: 归一化折损累计增益，衡量排序质量

---

## 2. 单粒度 vs 多粒度检索对比

| 指标 | 单粒度 | 多粒度 | 提升 |
|------|--------|--------|------|
| Recall@10 | {single.get('recall@10', 'N/A')} | {multi.get('recall@10', 'N/A')} | {improvement.get('recall@10', 'N/A')} |
| MRR | {single.get('mrr', 'N/A')} | {multi.get('mrr', 'N/A')} | {improvement.get('mrr', 'N/A')} |
| NDCG@10 | {single.get('ndcg@10', 'N/A')} | {multi.get('ndcg@10', 'N/A')} | {improvement.get('ndcg@10', 'N/A')} |

---

## 3. 各粒度贡献度分析

| 粒度 | 贡献度 | 说明 |
|------|--------|------|
| 类别级 | {contribution.get('category_contribution', 0.0):.2%} | 提供整体分类概览 |
| 观点级 | {contribution.get('aspect_contribution', 0.0):.2%} | 提供细粒度观点聚合 |
| 评论级 | {contribution.get('comment_contribution', 0.0):.2%} | 提供具体评论细节 |

---

## 4. 查询类型效果分析

| 查询类型 | 单粒度 Recall@10 | 多粒度 Recall@10 | 提升 |
|----------|-----------------|-----------------|------|
"""
        
        for query_type in ["general", "aspect", "specific"]:
            type_result = query_type_results.get(query_type, {})
            single_recall = type_result.get("single_granularity", {}).get("recall@10", "N/A")
            multi_recall = type_result.get("multi_granularity", {}).get("recall@10", "N/A")
            imp = type_result.get("improvement", {}).get("recall@10", "N/A")
            report += f"| {query_type} | {single_recall} | {multi_recall} | {imp} |\n"
        
        report += f"""
### 查询类型说明
- **泛化查询 (general)**: 询问酒店整体的、综合性的评价，如"酒店整体怎么样？"
- **观点查询 (aspect)**: 询问某个具体方面的评价，如"前台服务态度如何？"
- **具体查询 (specific)**: 询问非常具体、细节化的问题，如"泳池几点开放？"

---

## 5. 结论与建议

### 主要发现

"""
        
        # 自动生成结论
        if multi.get("recall@10", 0) > single.get("recall@10", 0):
            recall_diff = (multi.get("recall@10", 0) - single.get("recall@10", 0)) * 100
            report += f"1. **多粒度检索显著提升召回率**：相比单粒度，召回率提升 {recall_diff:.1f}%\n"
        else:
            report += "1. **单粒度检索在某些场景下表现更好**，建议针对不同查询类型选择合适的粒度\n"
        
        if multi.get("ndcg@10", 0) > single.get("ndcg@10", 0):
            ndcg_diff = (multi.get("ndcg@10", 0) - single.get("ndcg@10", 0)) * 100
            report += f"2. **多粒度检索改善排序质量**：NDCG@10 提升 {ndcg_diff:.1f}%\n"
        
        # 分析最佳粒度贡献
        contrib_sorted = sorted(
            [
                ("类别级", contribution.get("category_contribution", 0)),
                ("观点级", contribution.get("aspect_contribution", 0)),
                ("评论级", contribution.get("comment_contribution", 0))
            ],
            key=lambda x: x[1],
            reverse=True
        )
        report += f"3. **各粒度贡献排序**：{' > '.join([f"{name}({p:.1%})" for name, p in contrib_sorted])}\n"
        
        report += """
### 优化建议

"""
        
        # 根据分析生成建议
        for name, contrib in contrib_sorted[:2]:
            if contrib > 0.4:
                report += f"- **{name}贡献度较高**，建议继续优化该粒度的摘要质量\n"
        
        report += """
- 根据查询类型自动选择最优粒度组合
- 考虑引入查询意图识别来动态调整融合权重
- 定期评估各粒度贡献度，及时调整优化策略

---

*报告由 SummaryOptimizationEvaluator 自动生成*
"""
        
        return report
    
    def run_evaluation(
        self,
        test_queries: dict = None,
        ground_truth: dict = None
    ) -> dict:
        """
        运行完整评估流程
        
        Args:
            test_queries: 测试查询集（可选，默认使用TEST_QUERIES）
            ground_truth: 真实标签（可选）
        
        Returns:
            完整评估结果
        """
        if test_queries is None:
            test_queries = TEST_QUERIES
        
        # 合并所有查询
        all_queries = []
        for queries in test_queries.values():
            all_queries.extend(queries)
        
        # 如果没有提供ground truth，生成模拟数据
        if ground_truth is None:
            print("  警告: 未提供ground truth，使用模拟数据")
            ground_truth = self._generate_mock_ground_truth(all_queries)
        
        print(f"\n{'='*60}")
        print(f"开始评估")
        print(f"测试查询数: {len(all_queries)}")
        print(f"{'='*60}\n")
        
        # 1. 单粒度 vs 多粒度对比
        print("[1/3] 执行单粒度 vs 多粒度对比...")
        comparison = self.compare_single_vs_multi(all_queries, ground_truth)
        
        # 2. 查询类型效果分析
        print("[2/3] 分析查询类型效果...")
        query_type_effectiveness = self.analyze_query_type_effectiveness(
            test_queries, ground_truth
        )
        
        # 3. 粒度贡献度分析（使用多粒度检索结果）
        print("[3/3] 分析各粒度贡献度...")
        sample_results = []
        if self.adaptive_retriever:
            for query in all_queries[:3]:
                try:
                    results = self.adaptive_retriever.retrieve(query, topk=10)
                    sample_results.extend(results)
                except Exception:
                    pass
        elif self.indexer:
            for query in all_queries[:3]:
                try:
                    results = self.indexer.retrieve_with_fusion(query=query, topk=10)
                    sample_results.extend(results)
                except Exception:
                    pass
        
        granularity_contribution = self.analyze_granularity_contribution(sample_results)
        
        # 汇总结果
        evaluation_results = {
            "single_granularity": comparison["single_granularity"],
            "multi_granularity": comparison["multi_granularity"],
            "improvement": comparison["improvement"],
            "query_type_effectiveness": query_type_effectiveness,
            "granularity_contribution": granularity_contribution,
            "test_queries": test_queries,
            "ground_truth": ground_truth
        }
        
        # 保存评估历史
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": evaluation_results
        })
        
        print(f"\n{'='*60}")
        print(f"评估完成")
        print(f"{'='*60}")
        
        return evaluation_results
    
    def _generate_mock_ground_truth(self, queries: list) -> dict:
        """生成模拟的ground truth数据"""
        ground_truth = {}
        
        # 基于查询关键词生成模拟标签
        for query in queries:
            query_lower = query.lower()
            relevant_ids = []
            
            # 模拟生成相关ID
            if "服务" in query_lower or "前台" in query_lower:
                relevant_ids.extend([f"ASP_SVC_{i:03d}" for i in range(1, 6)])
            if "房间" in query_lower or "床" in query_lower:
                relevant_ids.extend([f"ASP_ROM_{i:03d}" for i in range(1, 6)])
            if "早餐" in query_lower:
                relevant_ids.extend([f"ASP_BRK_{i:03d}" for i in range(1, 6)])
            if "安静" in query_lower or "隔音" in query_lower:
                relevant_ids.extend([f"ASP_SND_{i:03d}" for i in range(1, 6)])
            
            # 如果没有匹配，添加一些默认ID
            if not relevant_ids:
                relevant_ids = [
                    f"ASP_SVC_{hash(query) % 10 + 1:03d}",
                    f"ASP_FAC_{hash(query) % 10 + 1:03d}",
                    f"CMT_{hash(query) % 100 + 1:05d}"
                ]
            
            ground_truth[query] = relevant_ids
        
        return ground_truth
    
    def save_evaluation_report(self, evaluation_results: dict, output_path: str):
        """
        保存评估报告
        
        Args:
            evaluation_results: 评估结果
            output_path: 输出文件路径
        """
        report = self.generate_report(evaluation_results)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"评估报告已保存: {output_path}")
    
    def save_evaluation_results(self, evaluation_results: dict, output_path: str):
        """
        保存评估结果（JSON格式）
        
        Args:
            evaluation_results: 评估结果
            output_path: 输出文件路径
        """
        # 移除不可序列化的部分
        serializable_results = {
            k: v for k, v in evaluation_results.items()
            if k not in ["ground_truth", "test_queries"]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存: {output_path}")


# ============== 使用示例 ==============

def demo_usage():
    """演示如何使用评估器"""
    
    # 模拟索引器和检索器
    class MockIndexer:
        def retrieve_with_fusion(self, query, topk=10, weights=None):
            # 模拟返回检索结果
            import random
            results = []
            for i in range(topk):
                results.append(MockResult(
                    item_id=f"CMT_{random.randint(1, 100):05d}",
                    item_type=random.choice(["category", "aspect", "comment"]),
                    text=f"模拟结果 {i+1}",
                    score=1.0 - (i * 0.1)
                ))
            return results
    
    class MockResult:
        def __init__(self, item_id, item_type, text, score):
            self.item_id = item_id
            self.item_type = item_type
            self.text = text
            self.score = score
    
    # 创建评估器
    indexer = MockIndexer()
    evaluator = SummaryOptimizationEvaluator(indexer=indexer)
    
    # 运行评估
    results = evaluator.run_evaluation()
    
    # 生成报告
    report = evaluator.generate_report(results)
    print("\n" + "="*60)
    print("评估报告预览:")
    print("="*60)
    print(report[:1000] + "...")
    
    return results


if __name__ == "__main__":
    demo_usage()
