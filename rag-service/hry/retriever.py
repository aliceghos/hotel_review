"""
检索器模块

支持五路检索 + 多模态检索的融合检索系统。

检索路径:
- FIRST_PATH: 类别级检索 (category)
- SECOND_PATH: 方面级检索 (aspect)
- THIRD_PATH: 评论级检索 (comment)
- FOURTH_PATH: 反向查询检索 (reverse_query)
- FIFTH_PATH: 多粒度摘要检索 (multi_granularity_summary)
- SIXTH_PATH: 多模态检索 (multimodal_retrieval)
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RetrievalPath(str, Enum):
    """检索路径枚举"""
    FIRST_PATH = "category_retrieval"           # 类别级
    SECOND_PATH = "aspect_retrieval"             # 方面级
    THIRD_PATH = "comment_retrieval"             # 评论级
    FOURTH_PATH = "reverse_query_retrieval"      # 反向查询
    FIFTH_PATH = "multi_granularity_summary"    # 多粒度摘要
    SIXTH_PATH = "multimodal_retrieval"         # 多模态检索


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 各路径权重
    path_weights: Dict[str, float] = None
    
    # 多模态配置
    multimodal_enabled: bool = True
    multimodal_fusion_weights: Dict[str, float] = None
    
    # 检索参数
    default_topk: int = 10
    category_topk: int = 5
    aspect_topk: int = 10
    comment_topk: int = 20
    reverse_query_topk: int = 20
    multimodal_topk: int = 10
    
    # RRF参数
    rrf_k: int = 60
    
    def __post_init__(self):
        if self.path_weights is None:
            self.path_weights = {
                "category": 0.15,
                "aspect": 0.20,
                "comment": 0.35,
                "reverse_query": 0.15,
                "multimodal": 0.15
            }
        
        if self.multimodal_fusion_weights is None:
            self.multimodal_fusion_weights = {"text": 0.7, "image": 0.3}


class BaseRetriever:
    """检索器基类"""
    
    def retrieve(self, query: str, topk: int = 10) -> List[Dict]:
        """检索方法，子类实现"""
        raise NotImplementedError
    
    def search(self, query: str, topk: int = 10) -> List[Dict]:
        """搜索方法别名"""
        return self.retrieve(query, topk)


class Retriever(BaseRetriever):
    """融合检索器
    
    支持六路检索的融合：
    1. 类别级检索
    2. 方面级检索
    3. 评论级检索
    4. 反向查询检索
    5. 多粒度摘要检索
    6. 多模态检索
    """
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        # 各检索器组件
        category_retriever: Optional[BaseRetriever] = None,
        aspect_retriever: Optional[BaseRetriever] = None,
        comment_retriever: Optional[BaseRetriever] = None,
        reverse_query_retriever: Optional[BaseRetriever] = None,
        multi_granularity_indexer: Optional[Any] = None,
        multimodal_retriever: Optional[Any] = None,
        # 回调函数
        intent_classifier: Optional[Callable] = None
    ):
        """
        初始化检索器
        
        Args:
            config: 检索配置
            category_retriever: 类别级检索器
            aspect_retriever: 方面级检索器
            comment_retriever: 评论级检索器
            reverse_query_retriever: 反向查询检索器
            multi_granularity_indexer: 多粒度索引器
            multimodal_retriever: 多模态检索器
            intent_classifier: 意图分类器
        """
        self.config = config or RetrievalConfig()
        
        # 各检索器组件
        self.category_retriever = category_retriever
        self.aspect_retriever = aspect_retriever
        self.comment_retriever = comment_retriever
        self.reverse_query_retriever = reverse_query_retriever
        self.multi_granularity_indexer = multi_granularity_indexer
        self.multimodal_retriever = multimodal_retriever
        self.intent_classifier = intent_classifier
        
        # 异步执行器
        self._executor = ThreadPoolExecutor(max_workers=6)
        
        # 日志
        self.retrieval_log: List[str] = []
        
        self._log("Retriever initialized")
    
    def _log(self, message: str):
        """记录日志"""
        logger.debug(message)
        self.retrieval_log.append(message)
    
    def retrieve(self, query: str, topk: int = 10, **kwargs) -> List[Dict]:
        """
        执行六路融合检索
        
        Args:
            query: 用户查询
            topk: 返回结果数量
            **kwargs: 其他参数
                - enable_multimodal: 是否启用多模态检索
                - include_images: 是否包含图片结果
                - intent: 预定义的意图类型
                
        Returns:
            检索结果列表
        """
        self.retrieval_log = []
        self._log(f"=== Retrieval started: '{query}' ===")
        
        enable_multimodal = kwargs.get("enable_multimodal", self.config.multimodal_enabled)
        include_images = kwargs.get("include_images", False)
        predefined_intent = kwargs.get("intent")
        
        # 意图分类
        intent_result = None
        if predefined_intent:
            intent_result = {"intent": predefined_intent, "confidence": 1.0}
        elif self.intent_classifier:
            intent_result = self._classify_intent(query)
        
        if intent_result:
            self._log(f"Intent: {intent_result.get('intent')}, confidence: {intent_result.get('confidence')}")
        
        # 检查是否需要多模态检索
        should_use_multimodal = False
        if enable_multimodal and self.multimodal_retriever:
            should_use_multimodal = self.multimodal_retriever.should_use_image_retrieval(query)
            self._log(f"Multimodal retrieval decision: {should_use_multimodal}")
        
        # 1-5路并行检索
        retrieval_tasks = [
            ("category", self._retrieve_category, query),
            ("aspect", self._retrieve_aspect, query),
            ("comment", self._retrieve_comment, query),
            ("reverse_query", self._retrieve_reverse_query, query),
            ("summary", self._retrieve_summary, query),
        ]
        
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(task[1], task[2]): task[0]
                for task in retrieval_tasks
            }
            for future in futures:
                path_name = futures[future]
                try:
                    results[path_name] = future.result()
                    self._log(f"{path_name}: {len(results[path_name])} results")
                except Exception as e:
                    self._log(f"{path_name} error: {e}")
                    results[path_name] = []
        
        # 第六路：多模态检索
        if should_use_multimodal and self.multimodal_retriever:
            self._log("Executing multimodal retrieval...")
            multimodal_results = self.multimodal_retriever.retrieve(
                query,
                include_images=include_images,
                topk=self.config.multimodal_topk
            )
            results["multimodal"] = multimodal_results
            self._log(f"multimodal: {len(multimodal_results)} results")
            
            # 合并多模态结果到评论
            if multimodal_results:
                results["comment"].extend(multimodal_results)
        else:
            results["multimodal"] = []
        
        # 融合各路结果
        fused = self._fuse_results(results, topk)
        
        # 添加元数据
        for r in fused:
            r["_retrieval_paths"] = list(results.keys())
            if intent_result:
                r["_intent"] = intent_result
        
        self._log(f"=== Retrieval completed: {len(fused)} results ===")
        
        return fused
    
    def _classify_intent(self, query: str) -> Dict:
        """意图分类"""
        if self.intent_classifier is None:
            return {"intent": "general", "confidence": 0.5}
        
        try:
            if hasattr(self.intent_classifier, 'classify_with_confidence'):
                return self.intent_classifier.classify_with_confidence(query)
            elif hasattr(self.intent_classifier, 'classify'):
                intent = self.intent_classifier.classify(query)
                return {"intent": intent, "confidence": 0.8}
        except Exception as e:
            self._log(f"Intent classification error: {e}")
        
        return {"intent": "general", "confidence": 0.5}
    
    def _retrieve_category(self, query: str) -> List[Dict]:
        """类别级检索"""
        if self.category_retriever is None:
            return []
        
        try:
            if hasattr(self.category_retriever, 'retrieve'):
                return self.category_retriever.retrieve(query, topk=self.config.category_topk)
            elif hasattr(self.category_retriever, 'search'):
                return self.category_retriever.search(query, topk=self.config.category_topk)
        except Exception as e:
            self._log(f"Category retrieval error: {e}")
        
        return []
    
    def _retrieve_aspect(self, query: str) -> List[Dict]:
        """方面级检索"""
        if self.aspect_retriever is None:
            return []
        
        try:
            if hasattr(self.aspect_retriever, 'retrieve'):
                return self.aspect_retriever.retrieve(query, topk=self.config.aspect_topk)
            elif hasattr(self.aspect_retriever, 'search'):
                return self.aspect_retriever.search(query, topk=self.config.aspect_topk)
        except Exception as e:
            self._log(f"Aspect retrieval error: {e}")
        
        return []
    
    def _retrieve_comment(self, query: str) -> List[Dict]:
        """评论级检索"""
        if self.comment_retriever is None:
            return []
        
        try:
            if hasattr(self.comment_retriever, 'retrieve'):
                return self.comment_retriever.retrieve(query, topk=self.config.comment_topk)
            elif hasattr(self.comment_retriever, 'search'):
                return self.comment_retriever.search(query, topk=self.config.comment_topk)
        except Exception as e:
            self._log(f"Comment retrieval error: {e}")
        
        return []
    
    def _retrieve_reverse_query(self, query: str) -> List[Dict]:
        """反向查询检索"""
        if self.reverse_query_retriever is None:
            return []
        
        try:
            if hasattr(self.reverse_query_retriever, 'retrieve'):
                return self.reverse_query_retriever.retrieve(query, topk=self.config.reverse_query_topk)
            elif hasattr(self.reverse_query_retriever, 'search'):
                return self.reverse_query_retriever.search(query, topk=self.config.reverse_query_topk)
        except Exception as e:
            self._log(f"Reverse query retrieval error: {e}")
        
        return []
    
    def _retrieve_summary(self, query: str) -> List[Dict]:
        """多粒度摘要检索"""
        if self.multi_granularity_indexer is None:
            return []
        
        try:
            if hasattr(self.multi_granularity_indexer, 'search'):
                return self.multi_granularity_indexer.search(
                    query,
                    level="summary",
                    topk=self.config.category_topk
                )
        except Exception as e:
            self._log(f"Summary retrieval error: {e}")
        
        return []
    
    def _fuse_results(self, results: Dict[str, List[Dict]], topk: int) -> List[Dict]:
        """
        融合各路检索结果
        
        Args:
            results: 各路径检索结果字典
            topk: 返回数量
            
        Returns:
            融合后的结果列表
        """
        path_weights = self.config.path_weights
        
        # 为每个结果添加来源权重
        fused_docs: Dict[str, Dict] = {}
        
        for path_name, path_results in results.items():
            weight = path_weights.get(path_name, 0.1)
            
            for rank, r in enumerate(path_results, start=1):
                doc_id = r.get("id", f"{path_name}_{rank}")
                
                if doc_id not in fused_docs:
                    fused_docs[doc_id] = {
                        "id": doc_id,
                        "content": r.get("content", r.get("comment", "")),
                        "score": 0.0,
                        "sources": [],
                        "metadata": {}
                    }
                
                # RRF分数
                rrf_score = 1.0 / (self.config.rrf_k + rank)
                
                # 基础分数
                base_score = r.get("score", r.get("relevance_score", 0.5))
                
                # 加权融合
                fused_docs[doc_id]["score"] += (rrf_score * 0.3 + base_score * 0.7) * weight
                fused_docs[doc_id]["sources"].append({
                    "path": path_name,
                    "rank": rank,
                    "score": base_score
                })
                
                # 合并元数据
                if "metadata" in r:
                    fused_docs[doc_id]["metadata"].update(r["metadata"])
        
        # 转换为列表并排序
        result_list = list(fused_docs.values())
        result_list.sort(key=lambda x: x["score"], reverse=True)
        
        # 去重并清理
        seen = set()
        final_results = []
        for r in result_list:
            content_key = hash(r.get("content", "")[:100])
            if content_key not in seen:
                seen.add(content_key)
                # 清理内部字段
                clean_r = {
                    "id": r["id"],
                    "content": r["content"],
                    "score": round(r["score"], 4),
                    "sources": r["sources"],
                    "metadata": r["metadata"]
                }
                final_results.append(clean_r)
        
        return final_results[:topk]
    
    def get_retrieval_log(self) -> List[str]:
        """获取检索日志"""
        return self.retrieval_log.copy()
    
    def get_retrieval_paths(self) -> List[str]:
        """获取支持的检索路径"""
        return [p.value for p in RetrievalPath]
    
    def is_multimodal_available(self) -> bool:
        """检查多模态检索是否可用"""
        return self.multimodal_retriever is not None
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class RetrieverFactory:
    """检索器工厂类"""
    
    @staticmethod
    def create_with_multimodal(
        text_retriever: Any,
        multimodal_retriever: Any,
        config: Optional[RetrievalConfig] = None
    ) -> Retriever:
        """
        创建带多模态检索的检索器
        
        Args:
            text_retriever: 文本检索器
            multimodal_retriever: 多模态检索器
            config: 检索配置
            
        Returns:
            配置好的检索器
        """
        return Retriever(
            config=config,
            comment_retriever=text_retriever,
            multimodal_retriever=multimodal_retriever
        )
    
    @staticmethod
    def create_from_adaptive_retriever(
        adaptive_retriever: Any,
        multimodal_retriever: Optional[Any] = None,
        config: Optional[RetrievalConfig] = None
    ) -> Retriever:
        """
        从自适应检索器创建融合检索器
        
        Args:
            adaptive_retriever: 自适应检索器
            multimodal_retriever: 多模态检索器
            config: 检索配置
            
        Returns:
            配置好的检索器
        """
        return Retriever(
            config=config,
            multi_granularity_indexer=getattr(adaptive_retriever, 'indexer', None),
            intent_classifier=getattr(adaptive_retriever, 'intent_classifier', None),
            multimodal_retriever=multimodal_retriever
        )


def create_default_retriever(
    multimodal_index_dir: Optional[str] = None,
    fusion_weights: Optional[Dict[str, float]] = None
) -> Retriever:
    """
    创建默认配置的检索器
    
    Args:
        multimodal_index_dir: 多模态索引目录
        fusion_weights: 融合权重
        
    Returns:
        默认检索器
    """
    from multimodal_retriever import create_multimodal_retriever
    
    # 创建多模态检索器
    multimodal = create_multimodal_retriever(
        image_index_dir=multimodal_index_dir,
        fusion_weights=fusion_weights
    )
    
    # 创建检索器
    config = RetrievalConfig()
    if fusion_weights:
        config.multimodal_fusion_weights = fusion_weights
    
    return Retriever(
        config=config,
        multimodal_retriever=multimodal
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建检索器
    retriever = create_default_retriever()
    
    print("=== Supported retrieval paths ===")
    for path in retriever.get_retrieval_paths():
        print(f"  - {path}")
    
    print("\n=== Multimodal available ===")
    print(retriever.is_multimodal_available())
    
    # 测试检索
    test_queries = [
        "酒店房间怎么样",
        "看看房间照片",
        "早餐好不好"
    ]
    
    print("\n=== Retrieval test ===")
    for q in test_queries:
        results = retriever.retrieve(q, topk=5)
        print(f"\nQuery: {q}")
        print(f"Results: {len(results)}")
        for r in results[:3]:
            print(f"  - score={r['score']:.3f}, content={r['content'][:50]}...")
