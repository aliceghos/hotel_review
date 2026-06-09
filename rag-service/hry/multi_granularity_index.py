"""
多粒度摘要索引模块

提供三级摘要的向量索引构建和检索功能：
- 类别级索引（关键词匹配）
- 观点级向量索引（ChromaDB）
- 评论级向量索引（ChromaDB）

融合策略：category_weight=0.2, aspect_weight=0.3, comment_weight=0.5
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


# ============== 数据结构 ==============

@dataclass
class AspectIndexItem:
    """观点级索引项"""
    aspect_id: str
    aspect_text: str
    aspect_detail: str
    keywords: List[str]
    category: str
    sentiment: str
    comment_count: int = 0


@dataclass
class CommentSummaryIndexItem:
    """评论级摘要索引项"""
    comment_id: str
    one_sentence_summary: str
    aspect_id: str
    category: str
    sentiment: str


@dataclass
class RetrievalResult:
    """检索结果"""
    item_id: str
    item_type: str  # "category" / "aspect" / "comment"
    text: str
    score: float
    metadata: Dict[str, Any]


# ============== 辅助函数 ==============

def _normalize_text(text: str) -> str:
    """标准化文本"""
    import re
    return re.sub(r"\s+", " ", text or "").strip()


def _generate_id(prefix: str, text: str) -> str:
    """生成哈希ID"""
    hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_val}"


def _keywords_match_score(query: str, keywords: List[str]) -> float:
    """计算关键词匹配分数"""
    if not keywords:
        return 0.0
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    keyword_words = set()
    
    for kw in keywords:
        keyword_words.update(kw.lower().split())
    
    overlap = query_words & keyword_words
    return len(overlap) / len(keyword_words) if keyword_words else 0.0


def _get_category_from_aspect_id(aspect_id: str) -> str:
    """从aspect_id提取类别"""
    parts = aspect_id.split("_")
    if len(parts) >= 2:
        abbrev = parts[1]
        # 反向映射
        abbreviations = {
            "SVC": "服务", "FAC": "设施", "BRK": "早餐", "ROM": "房间",
            "SND": "隔音", "HYG": "卫生", "LOC": "位置", "PRK": "停车",
            "ENV": "环境", "PRC": "价格", "DIN": "餐饮", "TRF": "交通",
            "SFT": "安全", "EXP": "体验",
        }
        return abbreviations.get(abbrev, "")
    return ""


# ============== 融合算法 ==============

def weighted_fusion(
    category_scores: Dict[str, float],
    aspect_scores: Dict[str, float],
    comment_scores: Dict[str, float],
    weights: Dict[str, float] = None
) -> List[Tuple[str, float, str]]:
    """
    多粒度检索结果加权融合
    
    Args:
        category_scores: 类别级检索结果 {category: score}
        aspect_scores: 观点级检索结果 {aspect_id: score}
        comment_scores: 评论级检索结果 {comment_id: score}
        weights: 各粒度权重
    
    Returns:
        融合后的排序结果列表 [(item_id, fused_score, item_type), ...]
    """
    if weights is None:
        weights = {"category": 0.2, "aspect": 0.3, "comment": 0.5}
    
    # 归一化分数
    def normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        min_score = min(scores.values())
        range_score = max_score - min_score if max_score != min_score else 1.0
        return {k: (v - min_score) / range_score for k, v in scores.items()}
    
    norm_category = normalize(category_scores)
    norm_aspect = normalize(aspect_scores)
    norm_comment = normalize(comment_scores)
    
    # 加权融合
    fused: Dict[str, float] = {}
    
    for cat, score in norm_category.items():
        fused[f"cat_{cat}"] = score * weights["category"]
    
    for aspect_id, score in norm_aspect.items():
        fused[f"aspect_{aspect_id}"] = score * weights["aspect"]
    
    for comment_id, score in norm_comment.items():
        fused[f"comment_{comment_id}"] = score * weights["comment"]
    
    # 排序返回
    sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    
    return [(item_id, score, item_id.split("_", 1)[0]) for item_id, score in sorted_results]


# ============== 主类实现 ==============

class MultiGranularityIndexer:
    """多粒度摘要索引构建器"""
    
    def __init__(
        self,
        embedding_client,
        chroma_client=None,
        persist_dir: str = None
    ):
        """
        初始化索引构建器
        
        Args:
            embedding_client: 文本嵌入客户端，需支持 embed(texts) -> List[List[float]]
            chroma_client: ChromaDB客户端（可选）
            persist_dir: 索引持久化目录
        """
        self.embedding_client = embedding_client
        self.persist_dir = persist_dir
        
        # ChromaDB客户端初始化
        if chroma_client is None and CHROMADB_AVAILABLE:
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(
                    path=str(Path(persist_dir) / "chroma_db"),
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self.chroma_client = chromadb.Client()
        else:
            self.chroma_client = chroma_client
        
        # 索引集合
        self.aspect_collection = None
        self.comment_collection = None
        
        # 内存索引缓存（用于不支持ChromaDB或增量更新）
        self._aspect_index_cache: Dict[str, AspectIndexItem] = {}
        self._comment_index_cache: Dict[str, CommentSummaryIndexItem] = {}
        
        # 已构建标记
        self._aspect_index_built = False
        self._comment_index_built = False
    
    def _ensure_chroma_available(self):
        """确保ChromaDB可用"""
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB未安装，请运行: pip install chromadb")
        if self.chroma_client is None:
            raise RuntimeError("ChromaDB客户端未初始化")
    
    # ============== 观点级索引构建 ==============
    
    def build_aspect_index(
        self,
        aspect_summaries: List[Dict],
        persist_dir: str = None
    ) -> bool:
        """
        构建观点级向量索引
        
        Args:
            aspect_summaries: 观点级摘要列表，每个字典包含：
                - aspect_id: 观点ID
                - aspect_text: 核心观点一句话描述
                - aspect_detail: 观点详细摘要
                - keywords: 关键词列表
                - category: 所属类别
                - sentiment: 情感倾向
                - comment_count: 评论数量
            persist_dir: 索引持久化目录
        
        Returns:
            是否构建成功
        """
        print(f"\n{'='*60}")
        print(f"构建观点级向量索引")
        print(f"观点数量: {len(aspect_summaries)}")
        print(f"{'='*60}")
        
        if not aspect_summaries:
            print("  警告: 无观点数据可索引")
            return False
        
        try:
            self._ensure_chroma_available()
            
            # 获取或创建collection
            collection_name = "aspect_summaries"
            try:
                self.aspect_collection = self.chroma_client.get_collection(
                    name=collection_name
                )
                # 清空旧数据
                self.chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass
            
            self.aspect_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "观点级摘要向量索引"}
            )
            
            # 准备索引数据
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, aspect in enumerate(aspect_summaries):
                aspect_id = aspect.get("aspect_id", f"ASP_{i+1:03d}")
                category = aspect.get("category", "")
                aspect_text = aspect.get("aspect_text", "")
                aspect_detail = aspect.get("aspect_detail", "")
                
                # 组合文本用于embedding
                combined_text = f"{aspect_text} {aspect_detail}"
                combined_text = _normalize_text(combined_text)
                
                # 生成embedding
                embedding = self.embedding_client.embed([combined_text])[0]
                
                # 存储
                ids.append(aspect_id)
                embeddings.append(embedding)
                metadatas.append({
                    "aspect_id": aspect_id,
                    "aspect_text": aspect_text[:100],
                    "aspect_detail": aspect_detail[:500],
                    "keywords": json.dumps(aspect.get("keywords", []), ensure_ascii=False),
                    "category": category,
                    "sentiment": aspect.get("sentiment", "neutral"),
                    "comment_count": aspect.get("comment_count", 0)
                })
                documents.append(combined_text)
                
                # 缓存
                self._aspect_index_cache[aspect_id] = AspectIndexItem(
                    aspect_id=aspect_id,
                    aspect_text=aspect_text,
                    aspect_detail=aspect_detail,
                    keywords=aspect.get("keywords", []),
                    category=category,
                    sentiment=aspect.get("sentiment", "neutral"),
                    comment_count=aspect.get("comment_count", 0)
                )
            
            # 批量添加
            self.aspect_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            self._aspect_index_built = True
            print(f"  观点级索引构建完成: {len(ids)} 条")
            
            # 持久化
            if persist_dir:
                self._persist_aspect_index(persist_dir)
            
            return True
            
        except Exception as e:
            print(f"  错误: 观点级索引构建失败: {e}")
            return False
    
    def _persist_aspect_index(self, persist_dir: str):
        """持久化观点级索引元数据"""
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # 保存缓存索引
        cache_file = persist_path / "aspect_index_cache.json"
        cache_data = {
            aspect_id: asdict(item) 
            for aspect_id, item in self._aspect_index_cache.items()
        }
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"  已持久化观点索引缓存: {cache_file}")
    
    def load_aspect_index(self, persist_dir: str) -> bool:
        """
        加载观点级索引
        
        Args:
            persist_dir: 索引持久化目录
        
        Returns:
            是否加载成功
        """
        persist_path = Path(persist_dir)
        cache_file = persist_path / "aspect_index_cache.json"
        
        if not cache_file.exists():
            print(f"  警告: 观点索引缓存文件不存在")
            return False
        
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            self._aspect_index_cache = {
                k: AspectIndexItem(**v) for k, v in cache_data.items()
            }
            
            print(f"  已加载观点索引缓存: {len(self._aspect_index_cache)} 条")
            return True
            
        except Exception as e:
            print(f"  错误: 加载观点索引失败: {e}")
            return False
    
    # ============== 评论级摘要索引构建 ==============
    
    def build_comment_summary_index(
        self,
        comment_summaries: List[Dict],
        persist_dir: str = None
    ) -> bool:
        """
        构建评论级摘要索引
        
        Args:
            comment_summaries: 评论级摘要列表，每个字典包含：
                - comment_id: 评论ID
                - one_sentence_summary: 一句话摘要
                - aspect_id: 关联的观点ID
                - category: 所属类别
                - sentiment: 情感倾向
            persist_dir: 索引持久化目录
        
        Returns:
            是否构建成功
        """
        print(f"\n{'='*60}")
        print(f"构建评论级摘要向量索引")
        print(f"评论数量: {len(comment_summaries)}")
        print(f"{'='*60}")
        
        if not comment_summaries:
            print("  警告: 无评论数据可索引")
            return False
        
        try:
            self._ensure_chroma_available()
            
            # 获取或创建collection
            collection_name = "comment_summaries"
            try:
                self.comment_collection = self.chroma_client.get_collection(
                    name=collection_name
                )
                self.chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass
            
            self.comment_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "评论级摘要向量索引"}
            )
            
            # 准备索引数据
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, comment in enumerate(comment_summaries):
                comment_id = comment.get("comment_id", f"CMT_{i+1:05d}")
                summary_text = comment.get("one_sentence_summary", "")
                
                if not summary_text:
                    continue
                
                summary_text = _normalize_text(summary_text)
                
                # 生成embedding
                embedding = self.embedding_client.embed([summary_text])[0]
                
                # 获取类别
                category = comment.get("category", "")
                if not category and comment.get("aspect_id"):
                    category = _get_category_from_aspect_id(comment.get("aspect_id", ""))
                
                # 存储
                ids.append(comment_id)
                embeddings.append(embedding)
                metadatas.append({
                    "comment_id": comment_id,
                    "one_sentence_summary": summary_text[:200],
                    "aspect_id": comment.get("aspect_id", ""),
                    "category": category,
                    "sentiment": comment.get("sentiment", "neutral")
                })
                documents.append(summary_text)
                
                # 缓存
                self._comment_index_cache[comment_id] = CommentSummaryIndexItem(
                    comment_id=comment_id,
                    one_sentence_summary=summary_text,
                    aspect_id=comment.get("aspect_id", ""),
                    category=category,
                    sentiment=comment.get("sentiment", "neutral")
                )
            
            # 批量添加
            self.comment_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            self._comment_index_built = True
            print(f"  评论级索引构建完成: {len(ids)} 条")
            
            # 持久化
            if persist_dir:
                self._persist_comment_index(persist_dir)
            
            return True
            
        except Exception as e:
            print(f"  错误: 评论级索引构建失败: {e}")
            return False
    
    def _persist_comment_index(self, persist_dir: str):
        """持久化评论级索引元数据"""
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # 保存缓存索引
        cache_file = persist_path / "comment_index_cache.json"
        cache_data = {
            comment_id: asdict(item)
            for comment_id, item in self._comment_index_cache.items()
        }
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"  已持久化评论索引缓存: {cache_file}")
    
    def load_comment_index(self, persist_dir: str) -> bool:
        """
        加载评论级索引
        
        Args:
            persist_dir: 索引持久化目录
        
        Returns:
            是否加载成功
        """
        persist_path = Path(persist_dir)
        cache_file = persist_path / "comment_index_cache.json"
        
        if not cache_file.exists():
            print(f"  警告: 评论索引缓存文件不存在")
            return False
        
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            self._comment_index_cache = {
                k: CommentSummaryIndexItem(**v) for k, v in cache_data.items()
            }
            
            print(f"  已加载评论索引缓存: {len(self._comment_index_cache)} 条")
            return True
            
        except Exception as e:
            print(f"  错误: 加载评论索引失败: {e}")
            return False
    
    # ============== 多粒度检索 ==============
    
    def _category_search(self, query: str, categories: List[str]) -> Dict[str, float]:
        """类别级检索（关键词匹配）"""
        scores = {}
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for category in categories:
            category_lower = category.lower()
            # 完全匹配
            if category_lower in query_lower:
                scores[category] = 1.0
            else:
                # 部分词匹配
                category_words = set(category_lower.split())
                overlap = query_words & category_words
                if overlap:
                    scores[category] = len(overlap) / len(category_words)
                else:
                    scores[category] = 0.0
        
        return scores
    
    def _aspect_search(self, query: str, topk: int = 10) -> Dict[str, float]:
        """观点级向量检索"""
        if not self._aspect_index_built or self.aspect_collection is None:
            return {}
        
        try:
            query_embedding = self.embedding_client.embed([query])[0]
            
            results = self.aspect_collection.query(
                query_embeddings=[query_embedding],
                n_results=topk
            )
            
            scores = {}
            if results and results.get("ids"):
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                
                # 距离转分数（假设余弦距离，越小越相似）
                for i, aspect_id in enumerate(ids):
                    distance = distances[i] if i < len(distances) else 1.0
                    # 转换为相似度分数
                    scores[aspect_id] = 1.0 / (1.0 + distance)
            
            return scores
            
        except Exception as e:
            print(f"  警告: 观点级检索失败: {e}")
            return {}
    
    def _comment_search(self, query: str, topk: int = 20) -> Dict[str, float]:
        """评论级向量检索"""
        if not self._comment_index_built or self.comment_collection is None:
            return {}
        
        try:
            query_embedding = self.embedding_client.embed([query])[0]
            
            results = self.comment_collection.query(
                query_embeddings=[query_embedding],
                n_results=topk
            )
            
            scores = {}
            if results and results.get("ids"):
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                
                for i, comment_id in enumerate(ids):
                    distance = distances[i] if i < len(distances) else 1.0
                    scores[comment_id] = 1.0 / (1.0 + distance)
            
            return scores
            
        except Exception as e:
            print(f"  警告: 评论级检索失败: {e}")
            return {}
    
    def retrieve_with_fusion(
        self,
        query: str,
        categories: List[str] = None,
        topk: int = 10,
        weights: Dict[str, float] = None
    ) -> List[RetrievalResult]:
        """
        多粒度检索融合
        
        融合策略：
        1. 类别级检索：根据query关键词匹配类别
        2. 观点级检索：语义匹配观点向量
        3. 评论级检索：语义匹配评论摘要向量
        
        Args:
            query: 查询文本
            categories: 可用的类别列表
            topk: 返回结果数量
            weights: 各粒度权重，默认 category=0.2, aspect=0.3, comment=0.5
        
        Returns:
            融合后的检索结果列表
        """
        if weights is None:
            weights = {"category": 0.2, "aspect": 0.3, "comment": 0.5}
        
        if categories is None:
            categories = list(set(item.category for item in self._aspect_index_cache.values()))
        
        # 1. 类别级检索
        category_scores = self._category_search(query, categories)
        
        # 2. 观点级检索
        aspect_scores = self._aspect_search(query, topk=20)
        
        # 3. 评论级检索
        comment_scores = self._comment_search(query, topk=50)
        
        # 4. 加权融合
        fused_results = weighted_fusion(
            category_scores, aspect_scores, comment_scores, weights
        )
        
        # 5. 转换为RetrievalResult并去重
        results = []
        seen_texts = set()
        seen_categories = set()
        
        for item_id, score, item_type in fused_results:
            if item_type == "cat":
                # 类别结果
                category = item_id.replace("cat_", "")
                if category in seen_categories:
                    continue
                seen_categories.add(category)
                
                results.append(RetrievalResult(
                    item_id=category,
                    item_type="category",
                    text=f"类别: {category}",
                    score=score,
                    metadata={"category": category}
                ))
                
            elif item_type == "aspect":
                # 观点结果
                aspect_id = item_id.replace("aspect_", "")
                cache_item = self._aspect_index_cache.get(aspect_id)
                if cache_item:
                    # 去重：相同文本的观点只返回一个
                    text_key = cache_item.aspect_text.lower()
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    
                    results.append(RetrievalResult(
                        item_id=aspect_id,
                        item_type="aspect",
                        text=cache_item.aspect_text,
                        score=score,
                        metadata=asdict(cache_item)
                    ))
                    
            elif item_type == "comment":
                # 评论结果
                comment_id = item_id.replace("comment_", "")
                cache_item = self._comment_index_cache.get(comment_id)
                if cache_item:
                    text_key = cache_item.one_sentence_summary.lower()
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    
                    results.append(RetrievalResult(
                        item_id=comment_id,
                        item_type="comment",
                        text=cache_item.one_sentence_summary,
                        score=score,
                        metadata=asdict(cache_item)
                    ))
            
            if len(results) >= topk:
                break
        
        return results
    
    # ============== 增量更新 ==============
    
    def add_aspect(self, aspect: Dict) -> bool:
        """
        增量添加观点
        
        Args:
            aspect: 观点数据字典
        
        Returns:
            是否添加成功
        """
        if self.aspect_collection is None:
            print("  错误: 观点索引未初始化")
            return False
        
        try:
            aspect_id = aspect.get("aspect_id")
            aspect_text = aspect.get("aspect_text", "")
            aspect_detail = aspect.get("aspect_detail", "")
            combined_text = _normalize_text(f"{aspect_text} {aspect_detail}")
            
            embedding = self.embedding_client.embed([combined_text])[0]
            
            self.aspect_collection.add(
                ids=[aspect_id],
                embeddings=[embedding],
                metadatas=[{
                    "aspect_id": aspect_id,
                    "aspect_text": aspect_text[:100],
                    "aspect_detail": aspect_detail[:500],
                    "keywords": json.dumps(aspect.get("keywords", []), ensure_ascii=False),
                    "category": aspect.get("category", ""),
                    "sentiment": aspect.get("sentiment", "neutral"),
                    "comment_count": aspect.get("comment_count", 0)
                }],
                documents=[combined_text]
            )
            
            # 更新缓存
            self._aspect_index_cache[aspect_id] = AspectIndexItem(
                aspect_id=aspect_id,
                aspect_text=aspect_text,
                aspect_detail=aspect_detail,
                keywords=aspect.get("keywords", []),
                category=aspect.get("category", ""),
                sentiment=aspect.get("sentiment", "neutral"),
                comment_count=aspect.get("comment_count", 0)
            )
            
            return True
            
        except Exception as e:
            print(f"  错误: 添加观点失败: {e}")
            return False
    
    def add_comment_summary(self, comment: Dict) -> bool:
        """
        增量添加评论摘要
        
        Args:
            comment: 评论摘要数据字典
        
        Returns:
            是否添加成功
        """
        if self.comment_collection is None:
            print("  错误: 评论索引未初始化")
            return False
        
        try:
            comment_id = comment.get("comment_id")
            summary_text = _normalize_text(comment.get("one_sentence_summary", ""))
            
            embedding = self.embedding_client.embed([summary_text])[0]
            
            category = comment.get("category", "")
            if not category and comment.get("aspect_id"):
                category = _get_category_from_aspect_id(comment.get("aspect_id", ""))
            
            self.comment_collection.add(
                ids=[comment_id],
                embeddings=[embedding],
                metadatas=[{
                    "comment_id": comment_id,
                    "one_sentence_summary": summary_text[:200],
                    "aspect_id": comment.get("aspect_id", ""),
                    "category": category,
                    "sentiment": comment.get("sentiment", "neutral")
                }],
                documents=[summary_text]
            )
            
            # 更新缓存
            self._comment_index_cache[comment_id] = CommentSummaryIndexItem(
                comment_id=comment_id,
                one_sentence_summary=summary_text,
                aspect_id=comment.get("aspect_id", ""),
                category=category,
                sentiment=comment.get("sentiment", "neutral")
            )
            
            return True
            
        except Exception as e:
            print(f"  错误: 添加评论摘要失败: {e}")
            return False
    
    # ============== 索引状态 ==============
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            "aspect_count": len(self._aspect_index_cache),
            "comment_count": len(self._comment_index_cache),
            "aspect_index_built": self._aspect_index_built,
            "comment_index_built": self._comment_index_built,
            "chroma_available": CHROMADB_AVAILABLE,
            "chroma_client_initialized": self.chroma_client is not None,
            "persist_dir": self.persist_dir
        }
    
    def rebuild_collections(self) -> bool:
        """重建ChromaDB collections"""
        if self.chroma_client is None:
            return False
        
        try:
            # 删除旧collections
            for name in ["aspect_summaries", "comment_summaries"]:
                try:
                    self.chroma_client.delete_collection(name=name)
                except Exception:
                    pass
            
            self.aspect_collection = None
            self.comment_collection = None
            self._aspect_index_built = False
            self._comment_index_built = False
            
            return True
            
        except Exception as e:
            print(f"  错误: 重建collections失败: {e}")
            return False


# ============== 使用示例 ==============

def demo_usage():
    """演示如何使用多粒度索引构建器"""
    
    # 模拟Embedding客户端
    class MockEmbeddingClient:
        def embed(self, texts: List[str]) -> List[List[float]]:
            # 返回随机embedding
            import random
            return [[random.random() for _ in range(10)] for _ in texts]
    
    # 创建索引器
    embedding_client = MockEmbeddingClient()
    indexer = MultiGranularityIndexer(
        embedding_client=embedding_client,
        persist_dir="output/indexes"
    )
    
    # 示例观点数据
    aspect_data = [
        {
            "aspect_id": "ASP_SVC_001",
            "aspect_text": "前台服务热情专业",
            "aspect_detail": "前台工作人员态度非常好，专业高效，办理入住和退房都很迅速。",
            "keywords": ["前台", "服务", "热情", "专业", "效率"],
            "category": "服务",
            "sentiment": "positive",
            "comment_count": 150
        },
        {
            "aspect_id": "ASP_FAC_001",
            "aspect_text": "房间设施齐全完好",
            "aspect_detail": "房间内设施齐全，空调、电视、热水器等都能正常工作。",
            "keywords": ["房间", "设施", "齐全", "完好"],
            "category": "设施",
            "sentiment": "positive",
            "comment_count": 120
        },
    ]
    
    # 示例评论数据
    comment_data = [
        {
            "comment_id": "CMT_00001",
            "one_sentence_summary": "前台服务非常热情，入住体验很好",
            "aspect_id": "ASP_SVC_001",
            "category": "服务",
            "sentiment": "positive"
        },
        {
            "comment_id": "CMT_00002",
            "one_sentence_summary": "房间很大，设施很新",
            "aspect_id": "ASP_FAC_001",
            "category": "设施",
            "sentiment": "positive"
        },
    ]
    
    # 构建索引
    print("\n=== 构建观点级索引 ===")
    indexer.build_aspect_index(aspect_data)
    
    print("\n=== 构建评论级索引 ===")
    indexer.build_comment_summary_index(comment_data)
    
    # 检索
    print("\n=== 多粒度检索 ===")
    results = indexer.retrieve_with_fusion(
        query="服务怎么样",
        categories=["服务", "设施", "房间", "卫生"],
        topk=5
    )
    
    print("\n检索结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.item_type}] {result.text} (score: {result.score:.4f})")
    
    # 索引统计
    print("\n=== 索引统计 ===")
    stats = indexer.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return indexer, results


if __name__ == "__main__":
    demo_usage()