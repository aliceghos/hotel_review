"""
多模态检索模块

支持文本检索、图像检索和跨模态检索的多模态检索系统。
集成到现有的检索框架中，提供第六路检索能力。
"""

import re
import logging
from typing import Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


class RetrievalPath(str, Enum):
    """检索路径枚举"""
    TEXT_ONLY = "text_retrieval"
    IMAGE_ONLY = "image_retrieval"
    MULTIMODAL_FUSED = "multimodal_retrieval"
    CROSS_MODAL = "cross_modal_retrieval"
    DIRECT_ANSWER = "direct_answer"


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class ImageRetrievalResult:
    """图像检索结果"""
    image_id: str
    image_path: str
    description: str
    score: float
    comment_id: Optional[str] = None
    comment_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "description": self.description,
            "score": self.score,
            "comment_id": self.comment_id,
            "comment_text": self.comment_text,
            "metadata": self.metadata
        }


class MultimodalRetriever:
    """多模态检索器
    
    支持文本检索、图像检索和跨模态检索的融合检索。
    可集成到现有的检索系统中作为第六路检索路径。
    """
    
    # 图像相关查询模式
    IMAGE_QUERY_PATTERNS = [
        r"照片", r"图片", r"图", r"看看",
        r"房间.*什么样", r"装修.*怎么样",
        r"设施.*图片", r"环境.*照片",
        r"长.*什么样", r"是.*什么样",
        r"看一下", r"瞧瞧", r"展示",
        r"房间.*照片", r"酒店.*外观",
        r"卫生间.*图", r"床.*图",
        r"有没有.*图", r"想看.*图"
    ]
    
    # 视觉相关主题关键词
    VISUAL_TOPICS = [
        "房间", "装修", "设施", "环境", "外观",
        "卫生间", "浴室", "床", "床垫", "枕头",
        "家具", "窗帘", "地毯", "灯光", "阳台",
        "景观", "海景", "城景", "花园", "泳池"
    ]
    
    def __init__(
        self,
        config: dict,
        text_retriever: Optional[object] = None,
        image_retriever: Optional[object] = None,
        multimodal_vectorizer: Optional[object] = None
    ):
        """
        初始化多模态检索器
        
        Args:
            config: 配置字典，包含:
                - fusion_weights: {"text": 0.7, "image": 0.3}
                - cross_modal_topk: 跨模态检索返回数量
                - image_cache_dir: 图像缓存目录
                - use_async: 是否使用异步检索
            text_retriever: 文本检索器实例
            image_retriever: 图像检索器实例
            multimodal_vectorizer: 多模态向量化器实例
        """
        self.config = config
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.vectorizer = multimodal_vectorizer
        
        # 融合权重
        self.fusion_weights = config.get("fusion_weights", {"text": 0.7, "image": 0.3})
        
        # 跨模态检索参数
        self.cross_modal_topk = config.get("cross_modal_topk", 10)
        
        # 缓存
        self._image_cache: Dict[str, List[ImageRetrievalResult]] = {}
        self._cache_enabled = config.get("cache_enabled", True)
        self._max_cache_size = config.get("max_cache_size", 1000)
        
        # 异步执行器
        self._use_async = config.get("use_async", True)
        self._executor = ThreadPoolExecutor(max_workers=4) if self._use_async else None
        
        # 日志
        self.retrieval_log: List[str] = []
        
        # RRF参数
        self.rrf_k = config.get("rrf_k", 60)
        
        self._log("MultimodalRetriever initialized")
        self._log(f"Fusion weights: {self.fusion_weights}")
    
    def _log(self, message: str):
        """记录日志"""
        logger.debug(message)
        self.retrieval_log.append(message)
    
    def should_use_image_retrieval(self, query: str) -> bool:
        """
        判断是否应该使用图片检索
        
        触发条件：
        - 查询包含"照片"、"图片"、"看看"、"视觉"等词
        - 查询涉及房间设施、装修等视觉相关主题
        
        Args:
            query: 用户查询
            
        Returns:
            是否应使用图像检索
        """
        query_lower = query.lower()
        
        # 检查图像查询模式
        for pattern in self.IMAGE_QUERY_PATTERNS:
            if re.search(pattern, query_lower):
                self._log(f"Query matched image pattern: {pattern}")
                return True
        
        # 检查视觉相关主题
        visual_topic_count = sum(1 for topic in self.VISUAL_TOPICS if topic in query_lower)
        if visual_topic_count >= 2:
            self._log(f"Query contains {visual_topic_count} visual topics")
            return True
        
        return False
    
    def is_image_related_query(self, query: str) -> bool:
        """
        判断是否为图像相关查询
        
        Args:
            query: 用户查询
            
        Returns:
            是否为图像相关查询
        """
        return self.should_use_image_retrieval(query)
    
    async def retrieve_async(
        self,
        query: str,
        include_images: bool = False,
        topk: int = 10
    ) -> List[Dict]:
        """
        异步多模态检索
        
        Args:
            query: 用户查询
            include_images: 是否返回图片检索结果
            topk: 返回结果数量
            
        Returns:
            检索结果列表
        """
        self._log(f"Async retrieve started: query='{query}', include_images={include_images}")
        
        # 判断是否使用图像检索
        use_image = self.should_use_image_retrieval(query)
        self._log(f"Image retrieval decision: {use_image}")
        
        if use_image and include_images:
            # 并行执行文本和图像检索
            text_task = asyncio.create_task(
                self._text_retrieve_async(query, topk)
            )
            image_task = asyncio.create_task(
                self._image_retrieve_async(query, self.cross_modal_topk)
            )
            
            text_results, image_results = await asyncio.gather(text_task, image_task)
            
            # 融合结果
            fused = self.fuse_results(
                text_results,
                image_results,
                self.fusion_weights.get("text", 0.7),
                self.fusion_weights.get("image", 0.3)
            )
            
            # 添加图像结果到元数据
            for r in fused:
                r["_retrieval_path"] = RetrievalPath.MULTIMODAL_FUSED
                if image_results:
                    r["_image_results"] = [img.to_dict() for img in image_results[:3]]
            
            return fused[:topk]
        else:
            # 仅文本检索
            return await self._text_retrieve_async(query, topk)
    
    def retrieve(
        self,
        query: str,
        include_images: bool = False,
        topk: int = 10
    ) -> List[Dict]:
        """
        多模态检索
        
        Args:
            query: 用户查询
            include_images: 是否返回图片检索结果
            topk: 返回结果数量
            
        Returns:
            检索结果列表
        """
        self._log(f"Retrieve started: query='{query}', include_images={include_images}")
        
        # 判断是否使用图像检索
        use_image = self.should_use_image_retrieval(query)
        self._log(f"Image retrieval decision: {use_image}")
        
        if use_image and include_images:
            # 同步并行执行
            with ThreadPoolExecutor(max_workers=2) as executor:
                text_future = executor.submit(self._text_retrieve, query, topk)
                image_future = executor.submit(self._image_retrieve, query, self.cross_modal_topk)
                
                text_results = text_future.result()
                image_results = image_future.result()
            
            # 融合结果
            fused = self.fuse_results(
                text_results,
                image_results,
                self.fusion_weights.get("text", 0.7),
                self.fusion_weights.get("image", 0.3)
            )
            
            # 添加检索路径信息
            for r in fused:
                r["_retrieval_path"] = RetrievalPath.MULTIMODAL_FUSED
                if image_results:
                    r["_image_results"] = [img.to_dict() for img in image_results[:3]]
            
            return fused[:topk]
        else:
            # 仅文本检索
            return self._text_retrieve(query, topk)
    
    def _text_retrieve(self, query: str, topk: int) -> List[Dict]:
        """文本检索"""
        if self.text_retriever is None:
            self._log("Text retriever not available, returning empty")
            return []
        
        try:
            if hasattr(self.text_retriever, 'retrieve'):
                results = self.text_retriever.retrieve(query, topk=topk)
            elif hasattr(self.text_retriever, 'search'):
                results = self.text_retriever.search(query, topk=topk)
            else:
                self._log("Text retriever has no retrieve/search method")
                return []
            
            # 转换为统一格式
            formatted = []
            for r in results:
                if isinstance(r, dict):
                    formatted.append({
                        "id": r.get("id", ""),
                        "content": r.get("content", r.get("comment", r.get("text", ""))),
                        "score": r.get("score", r.get("relevance_score", 0.5)),
                        "source": "text",
                        "metadata": {k: v for k, v in r.items() if k not in ["id", "content", "score"]}
                    })
            
            self._log(f"Text retrieval returned {len(formatted)} results")
            return formatted
            
        except Exception as e:
            self._log(f"Text retrieval error: {e}")
            return []
    
    async def _text_retrieve_async(self, query: str, topk: int) -> List[Dict]:
        """异步文本检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._text_retrieve, query, topk)
    
    def _image_retrieve(self, query: str, topk: int) -> List[ImageRetrievalResult]:
        """图像检索"""
        # 检查缓存
        cache_key = self._get_cache_key(query, "image")
        if cache_key in self._image_cache:
            self._log(f"Image results retrieved from cache")
            return self._image_cache[cache_key][:topk]
        
        if self.image_retriever is None and self.vectorizer is None:
            self._log("No image retriever or vectorizer available")
            return []
        
        try:
            if self.vectorizer is not None:
                results = self._cross_modal_retrieve_impl(query, topk)
            else:
                self._log("No vectorizer for cross-modal retrieval")
                results = []
            
            # 更新缓存
            if results and self._cache_enabled:
                self._update_cache(cache_key, results)
            
            return results
            
        except Exception as e:
            self._log(f"Image retrieval error: {e}")
            return []
    
    async def _image_retrieve_async(self, query: str, topk: int) -> List[ImageRetrievalResult]:
        """异步图像检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._image_retrieve, query, topk)
    
    def cross_modal_retrieve(self, query: str, topk: int = 10) -> List[Dict]:
        """
        文本到图像跨模态检索
        
        1. 将文本查询编码为向量
        2. 在图片向量库中检索相似图片
        3. 返回图片及对应评论
        
        Args:
            query: 用户查询
            topk: 返回结果数量
            
        Returns:
            检索结果列表，每项包含图片和评论信息
        """
        self._log(f"Cross-modal retrieval: query='{query}', topk={topk}")
        
        results = self._cross_modal_retrieve_impl(query, topk)
        
        # 转换为字典格式
        formatted = []
        for r in results:
            item = {
                "image_id": r.image_id,
                "image_path": r.image_path,
                "description": r.description,
                "score": r.score,
                "comment_id": r.comment_id,
                "comment_text": r.comment_text,
                "source": "cross_modal",
                "metadata": r.metadata
            }
            formatted.append(item)
        
        self._log(f"Cross-modal retrieval returned {len(formatted)} results")
        return formatted
    
    def _cross_modal_retrieve_impl(
        self,
        query: str,
        topk: int
    ) -> List[ImageRetrievalResult]:
        """跨模态检索实现"""
        if self.vectorizer is None:
            return []
        
        try:
            # 1. 将文本查询编码为向量
            query_vector = self.vectorizer.encode_text(query)
            if query_vector is None:
                self._log("Failed to encode query")
                return []
            
            # 2. 在图片向量库中检索相似图片
            # 假设向量库数据存储在配置中
            index_data = self._get_image_index_data()
            if index_data is None or len(index_data.get("image_vectors", [])) == 0:
                self._log("No image index data available")
                return []
            
            # 计算相似度
            similarities = []
            image_vectors = index_data["image_vectors"]
            
            for i, image_vec in enumerate(image_vectors):
                score = self.vectorizer.get_vector_similarity(
                    query_vector,
                    image_vec,
                    method="cosine"
                )
                similarities.append({
                    "index": i,
                    "score": float(score)
                })
            
            # 排序并返回topk
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # 构建结果
            results = []
            metadata = index_data.get("metadata", {})
            image_paths = metadata.get("image_paths", [])
            descriptions = metadata.get("descriptions", [])
            comment_ids = index_data.get("comment_ids", [])
            comment_texts = index_data.get("comment_texts", [])
            
            for item in similarities[:topk]:
                idx = item["index"]
                results.append(ImageRetrievalResult(
                    image_id=f"img_{idx}",
                    image_path=image_paths[idx] if idx < len(image_paths) else "",
                    description=descriptions[idx] if idx < len(descriptions) else "",
                    score=item["score"],
                    comment_id=comment_ids[idx] if idx < len(comment_ids) else None,
                    comment_text=comment_texts[idx] if idx < len(comment_texts) else None,
                    metadata={"rank": idx + 1}
                ))
            
            return results
            
        except Exception as e:
            self._log(f"Cross-modal retrieval error: {e}")
            return []
    
    def _get_image_index_data(self) -> Optional[dict]:
        """获取图像索引数据"""
        # 优先使用配置中的索引数据
        if hasattr(self, '_image_index_data'):
            return self._image_index_data
        
        # 尝试从向量化器加载
        if self.vectorizer is not None and hasattr(self.vectorizer, 'load_index'):
            try:
                index_dir = self.config.get("image_index_dir")
                if index_dir:
                    self._image_index_data = self.vectorizer.load_index(index_dir)
                    return self._image_index_data
            except Exception as e:
                self._log(f"Failed to load image index: {e}")
        
        return None
    
    def set_image_index_data(self, index_data: dict):
        """设置图像索引数据"""
        self._image_index_data = index_data
        self._log(f"Image index data set with {len(index_data.get('comment_ids', []))} images")
    
    def fuse_results(
        self,
        text_results: List[Dict],
        image_results: List[ImageRetrievalResult],
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[Dict]:
        """
        文本和图像检索结果融合
        
        融合策略：
        1. RRF (Reciprocal Rank Fusion)
        2. 分数加权融合
        3. 稀释融合 (当某一路结果明显较好时降低另一路权重)
        
        Args:
            text_results: 文本检索结果
            image_results: 图像检索结果
            text_weight: 文本权重
            image_weight: 图像权重
            
        Returns:
            融合后的结果列表
        """
        self._log(f"Fusing results: text={len(text_results)}, image={len(image_results)}")
        
        if not text_results and not image_results:
            return []
        
        if not text_results:
            return self._format_image_results(image_results)
        
        if not image_results:
            return text_results
        
        # 计算质量分数用于稀释融合
        text_quality = self._calculate_result_quality(text_results)
        image_quality = self._calculate_result_quality(
            [{"score": r.score} for r in image_results]
        )
        
        # 稀释因子：当一路结果明显较好时降低另一路权重
        dilution_factor = self._calculate_dilution(text_quality, image_quality)
        
        adjusted_text_weight = text_weight * (1 - dilution_factor * image_weight)
        adjusted_image_weight = image_weight * (1 - dilution_factor * text_weight)
        
        # 归一化
        total = adjusted_text_weight + adjusted_image_weight
        if total > 0:
            adjusted_text_weight /= total
            adjusted_image_weight /= total
        
        self._log(f"Adjusted weights: text={adjusted_text_weight:.3f}, image={adjusted_image_weight:.3f}")
        
        # 使用RRF融合
        fused = self._rrf_fusion(text_results, image_results)
        
        # 应用权重调整
        for r in fused:
            if r.get("source") == "text":
                r["score"] *= adjusted_text_weight
            else:
                r["score"] *= adjusted_image_weight
        
        # 重新排序
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        return fused
    
    def _calculate_result_quality(self, results: List[Dict]) -> float:
        """计算结果质量分数"""
        if not results:
            return 0.0
        
        # 基于平均分数和结果数量
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        count_bonus = min(len(results) / 10, 1.0) * 0.2
        
        return min(1.0, avg_score + count_bonus)
    
    def _calculate_dilution(self, quality_a: float, quality_b: float) -> float:
        """计算稀释因子"""
        if quality_a == 0 or quality_b == 0:
            return 0.0
        
        ratio = max(quality_a, quality_b) / max(min(quality_a, quality_b), 0.01)
        
        if ratio > 3.0:
            return 0.5
        elif ratio > 2.0:
            return 0.3
        else:
            return 0.0
    
    def _rrf_fusion(
        self,
        text_results: List[Dict],
        image_results: List[ImageRetrievalResult]
    ) -> List[Dict]:
        """
        使用RRF (Reciprocal Rank Fusion) 融合结果
        
        Args:
            text_results: 文本检索结果
            image_results: 图像检索结果
            
        Returns:
            融合后的结果
        """
        fused_scores: Dict[str, Dict] = {}
        
        # 文本结果RRF
        for rank, r in enumerate(text_results, start=1):
            doc_id = r.get("id", f"text_{rank}")
            fused_scores[doc_id] = {
                "id": doc_id,
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "source": "text",
                "metadata": r.get("metadata", {})
            }
            fused_scores[doc_id]["_rrf_score"] = 1.0 / (self.rrf_k + rank)
        
        # 图像结果RRF (基于关联的评论)
        for rank, r in enumerate(image_results, start=1):
            # 使用评论ID作为融合依据
            ref_id = r.comment_id or r.image_id
            if not ref_id:
                ref_id = f"image_{rank}"
            
            if ref_id in fused_scores:
                # 已存在的文档，增加RRF分数
                fused_scores[ref_id]["_rrf_score"] += 1.0 / (self.rrf_k + rank)
                fused_scores[ref_id]["score"] += r.score * 0.5
                # 添加图像信息到元数据
                if "related_images" not in fused_scores[ref_id]["metadata"]:
                    fused_scores[ref_id]["metadata"]["related_images"] = []
                fused_scores[ref_id]["metadata"]["related_images"].append(r.to_dict())
            else:
                # 新文档
                fused_scores[ref_id] = {
                    "id": ref_id,
                    "content": r.comment_text or r.description or "",
                    "score": r.score,
                    "source": "image",
                    "metadata": {
                        "related_images": [r.to_dict()]
                    }
                }
                fused_scores[ref_id]["_rrf_score"] = 1.0 / (self.rrf_k + rank)
        
        # 合并RRF分数和原始分数
        result_list = []
        for doc_id, data in fused_scores.items():
            final_score = (
                data["_rrf_score"] * 0.6 +
                data["score"] * 0.4
            )
            result_list.append({
                "id": data["id"],
                "content": data["content"],
                "score": final_score,
                "source": data["source"],
                "metadata": data["metadata"]
            })
        
        result_list.sort(key=lambda x: x["score"], reverse=True)
        return result_list
    
    def _format_image_results(
        self,
        image_results: List[ImageRetrievalResult]
    ) -> List[Dict]:
        """将图像结果格式化为标准输出格式"""
        formatted = []
        for r in image_results:
            content = r.comment_text or r.description or ""
            if not content:
                continue
            formatted.append({
                "id": r.comment_id or r.image_id,
                "content": content,
                "score": r.score,
                "source": "image",
                "metadata": {
                    "image_path": r.image_path,
                    "image_description": r.description,
                    "related_images": [r.to_dict()]
                }
            })
        return formatted
    
    def _get_cache_key(self, query: str, retrieval_type: str) -> str:
        """生成缓存键"""
        key_str = f"{retrieval_type}:{query}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_cache(self, cache_key: str, results: List):
        """更新缓存"""
        if len(self._image_cache) >= self._max_cache_size:
            # FIFO淘汰
            oldest_key = next(iter(self._image_cache))
            del self._image_cache[oldest_key]
        
        self._image_cache[cache_key] = results
    
    def clear_cache(self):
        """清空缓存"""
        self._image_cache.clear()
        self._log("Cache cleared")
    
    def get_retrieval_log(self) -> List[str]:
        """获取检索日志"""
        return self.retrieval_log.copy()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "text_retriever": self.text_retriever is not None,
            "image_retriever": self.image_retriever is not None,
            "vectorizer": self.vectorizer is not None,
            "fusion_weights": self.fusion_weights,
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._image_cache)
        }
    
    def __del__(self):
        """析构函数"""
        if self._executor:
            self._executor.shutdown(wait=False)


class MultimodalRetrieverFactory:
    """多模态检索器工厂类"""
    
    @staticmethod
    def create_from_config(config: dict) -> MultimodalRetriever:
        """
        从配置创建多模态检索器
        
        Args:
            config: 配置字典
            
        Returns:
            配置好的多模态检索器实例
        """
        from multimodal_vectorizer import MultimodalVectorizer
        from image_enhancement import ImageEnhancer
        
        # 创建向量化器
        vectorizer = None
        if config.get("vectorizer", {}).get("enabled", True):
            try:
                vectorizer = MultimodalVectorizer(config.get("vectorizer", {}))
            except Exception as e:
                logger.warning(f"Failed to create vectorizer: {e}")
        
        # 创建图片增强器（用于图像检索）
        image_enhancer = None
        if config.get("image_enhancer", {}).get("enabled", True):
            try:
                image_enhancer = ImageEnhancer(config.get("image_enhancer", {}))
            except Exception as e:
                logger.warning(f"Failed to create image enhancer: {e}")
        
        return MultimodalRetriever(
            config=config,
            multimodal_vectorizer=vectorizer,
            image_retriever=image_enhancer
        )


def create_multimodal_retriever(
    text_retriever: Optional[object] = None,
    image_index_dir: Optional[str] = None,
    fusion_weights: Optional[dict] = None,
    device: str = "cpu"
) -> MultimodalRetriever:
    """
    创建多模态检索器的便捷函数
    
    Args:
        text_retriever: 文本检索器实例
        image_index_dir: 图像索引目录
        fusion_weights: 融合权重
        device: 设备类型
        
    Returns:
        多模态检索器实例
    """
    from multimodal_vectorizer import MultimodalVectorizer
    
    config = {
        "fusion_weights": fusion_weights or {"text": 0.7, "image": 0.3},
        "cross_modal_topk": 10,
        "cache_enabled": True,
        "max_cache_size": 1000,
        "image_index_dir": image_index_dir,
        "vectorizer": {
            "model_type": "clip",
            "device": device,
            "batch_size": 8
        }
    }
    
    vectorizer = None
    try:
        vectorizer = MultimodalVectorizer(config["vectorizer"])
    except Exception as e:
        logger.warning(f"Failed to initialize vectorizer: {e}")
    
    return MultimodalRetriever(
        config=config,
        text_retriever=text_retriever,
        vectorizer=vectorizer
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建检索器
    retriever = MultimodalRetriever(
        config={
            "fusion_weights": {"text": 0.7, "image": 0.3},
            "cross_modal_topk": 10
        }
    )
    
    # 测试图像查询判断
    test_queries = [
        "酒店房间照片",
        "房间装修怎么样",
        "酒店服务好不好",
        "看看房间设施图片",
        "早餐怎么样"
    ]
    
    print("\n=== 图像查询判断测试 ===")
    for q in test_queries:
        result = retriever.should_use_image_retrieval(q)
        print(f"'{q}' -> {result}")
    
    print("\n=== 模型信息 ===")
    print(retriever.get_model_info())
