"""
自适应检索系统

根据用户查询意图自动调整检索策略，动态调整各粒度摘要的权重。
"""

import re
import json
from typing import Optional, Callable


INTENT_CLASSIFICATION_PROMPT = """
你是一个酒店问答系统的意图分析专家。请分析以下用户查询的意图类型。

【查询】
{query}

【意图类型定义】
1. general (泛化查询): 询问酒店整体的、综合性的评价或体验
   特征：使用"怎么样"、"好吗"、"如何"等模糊词
   示例："酒店整体怎么样？"、"服务好吗？"

2. aspect (观点查询): 询问酒店某个具体方面的观点或评价
   特征：涉及特定的服务或设施类别
   示例："前台服务态度如何？"、"房间安静吗？"、"早餐丰富吗？"

3. specific (具体查询): 询问非常具体、细节化的问题
   特征：涉及具体房型、设施细节、时间等
   示例："红棉大床房的床垫硬度如何？"、"泳池开放到几点？"

【输出格式】
仅输出JSON格式：
{{
    "intent": "general" | "aspect" | "specific",
    "confidence": 0.0-1.0,
    "reasoning": "分类理由（30字以内）"
}}
"""


class QueryIntentClassifier:
    """查询意图分类器"""
    
    INTENT_TYPES = ["general", "aspect", "specific"]
    """
    - general: 泛化查询，如"酒店整体怎么样？"、"服务好吗？"
    - aspect: 观点查询，如"前台服务态度如何？"、"房间安静吗？"
    - specific: 具体查询，如"红棉大床房的床品舒适度"、"泳池几点开放？"
    """
    
    # 规则模式
    GENERAL_PATTERNS = [
        r'怎么样', r'如何', r'好吗', r'如何', r'整体.*怎么样',
        r'总体.*如何', r'大概.*怎样', r'评价.*怎么样'
    ]
    
    SPECIFIC_PATTERNS = [
        r'房', r'床', r'池', r'开放.*点', r'几点',
        r'硬度', r'温度', r'品牌', r'型号', r'规格',
        r'(?:房型|房间).*(?:大小|面积|床型)',
    ]
    
    ASPECT_PATTERNS = [
        r'服务', r'态度', r'卫生', r'干净', r'安静',
        r'早餐', r'晚餐', r'设施', r'环境', r'交通',
        r'价格', r'性价比', r'位置', r'前台', r'餐饮'
    ]
    
    def __init__(self, llm_callable: Optional[Callable] = None):
        """
        初始化意图分类器
        
        Args:
            llm_callable: LLM调用函数，签名同上
        """
        self.llm_callable = llm_callable
    
    def classify(self, query: str) -> str:
        """
        分类用户查询
        
        Returns:
            "general" | "aspect" | "specific"
        """
        result = self.classify_with_confidence(query)
        return result["intent"]
    
    def classify_with_confidence(self, query: str) -> dict:
        """
        带置信度的分类
        
        Returns:
            {
                "intent": str,
                "confidence": float,
                "reasoning": str
            }
        """
        # 优先使用LLM分类
        if self.llm_callable:
            try:
                return self._classify_by_llm(query)
            except Exception:
                pass
        
        # 回退到规则分类
        return self._classify_by_rules(query)
    
    def _classify_by_llm(self, query: str) -> dict:
        """使用LLM进行意图分类"""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        response = self.llm_callable(prompt)
        
        # 解析JSON响应
        try:
            result = json.loads(response)
            return {
                "intent": result.get("intent", "general"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
        except (json.JSONDecodeError, ValueError):
            # 解析失败，回退到规则
            return self._classify_by_rules(query)
    
    def _classify_by_rules(self, query: str) -> dict:
        """基于规则的意图分类"""
        scores = {"general": 0.0, "aspect": 0.0, "specific": 0.0}
        
        query_lower = query.lower()
        
        # 检查general模式
        for pattern in self.GENERAL_PATTERNS:
            if re.search(pattern, query_lower):
                scores["general"] += 0.8
        
        # 检查specific模式
        for pattern in self.SPECIFIC_PATTERNS:
            if re.search(pattern, query_lower):
                scores["specific"] += 0.6
        
        # 检查aspect模式
        for pattern in self.ASPECT_PATTERNS:
            if re.search(pattern, query_lower):
                scores["aspect"] += 0.5
        
        # 查询长度特征
        query_len = len(query)
        if query_len < 10:
            scores["general"] += 0.3
        elif query_len > 20:
            scores["specific"] += 0.2
        
        # 具体房型/设施检测
        room_type_patterns = [
            r'(?:红棉|豪华|标准|套房|单人|双人|大床|双床)',
            r'(?:海景|城景|山景|湖景)',
        ]
        for pattern in room_type_patterns:
            if re.search(pattern, query):
                scores["specific"] += 0.4
        
        # 时间相关检测
        time_patterns = [r'\d+点', r'几点', r'开放', r'营业', r'入住', r'退房']
        for pattern in time_patterns:
            if re.search(pattern, query_lower):
                scores["specific"] += 0.5
        
        # 程度词检测
        degree_words = [r'非常', r'特别', r'十分', r'相当']
        for word in degree_words:
            if word in query_lower:
                scores["aspect"] += 0.2
        
        # 选择最高分
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]
        
        # 计算置信度
        if max_score == 0:
            confidence = 0.3
            max_intent = "general"
            reasoning = "默认泛化查询"
        else:
            total = sum(scores.values())
            confidence = max_score / total if total > 0 else 0.3
            # 归一化置信度
            confidence = min(0.95, max(0.4, confidence))
            
            reasoning_map = {
                "general": "包含泛化特征词",
                "aspect": "涉及具体方面",
                "specific": "包含具体细节"
            }
            reasoning = reasoning_map[max_intent]
        
        return {
            "intent": max_intent,
            "confidence": round(confidence, 2),
            "reasoning": reasoning
        }


class AdaptiveRetriever:
    """自适应检索器"""
    
    DEFAULT_WEIGHTS = {
        "general": {"category": 0.6, "aspect": 0.3, "comment": 0.1},
        "aspect": {"category": 0.2, "aspect": 0.5, "comment": 0.3},
        "specific": {"category": 0.1, "aspect": 0.2, "comment": 0.7}
    }
    
    def __init__(self, multi_granularity_indexer, intent_classifier):
        """
        初始化自适应检索器
        
        Args:
            multi_granularity_indexer: 多粒度索引检索器
            intent_classifier: 意图分类器
        """
        self.indexer = multi_granularity_indexer
        self.intent_classifier = intent_classifier
        self.retrieval_log = []
    
    def retrieve(self, query: str, topk: int = 10, custom_weights: dict = None) -> list[dict]:
        """
        自适应检索
        
        1. 意图分类
        2. 选择权重配置
        3. 多粒度检索
        4. 加权融合
        
        Args:
            query: 用户查询
            topk: 返回结果数量
            custom_weights: 自定义权重，会覆盖默认权重
        
        Returns:
            检索结果列表
        """
        self.retrieval_log = []
        self._log(f"开始检索: {query}")
        
        # 1. 意图分类
        intent_result = self.intent_classifier.classify_with_confidence(query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        self._log(f"意图分类: {intent}, 置信度: {confidence}, 理由: {intent_result['reasoning']}")
        
        # 2. 获取权重配置
        if custom_weights:
            weights = custom_weights
            self._log(f"使用自定义权重: {weights}")
        else:
            weights = self.DEFAULT_WEIGHTS[intent].copy()
            self._log(f"使用默认权重(意图={intent}): {weights}")
        
        # 3. 动态权重调整
        query_features = self._extract_query_features(query)
        weights = self.adjust_weights(weights, query_features)
        self._log(f"调整后权重: {weights}")
        
        # 4. 多粒度检索
        self._log("执行多粒度检索...")
        
        category_results = self.indexer.search(query, level="category", topk=topk)
        aspect_results = self.indexer.search(query, level="aspect", topk=topk)
        comment_results = self.indexer.search(query, level="comment", topk=topk)
        
        self._log(f"类别级检索: {len(category_results)} 条, 方面级检索: {len(aspect_results)} 条, 评论级检索: {len(comment_results)} 条")
        
        # 5. 加权融合
        fused_results = self._fuse_results(
            category_results, aspect_results, comment_results,
            weights, topk
        )
        
        self._log(f"融合完成，返回 {len(fused_results)} 条结果")
        
        # 添加检索路径信息
        for i, result in enumerate(fused_results):
            result["_retrieval_path"] = {
                "intent": intent,
                "confidence": confidence,
                "weights": weights,
                "query_features": query_features
            }
        
        return fused_results
    
    def _extract_query_features(self, query: str) -> dict:
        """
        提取查询特征
        
        Returns:
            {
                "has_time_sensitivity": bool,  # 是否时间敏感
                "has_room_type": bool,        # 是否指定房型
                "query_length": int,          # 查询长度
                "specificity_score": float     # 具体性得分
            }
        """
        query_lower = query.lower()
        
        # 时间敏感性检测
        time_keywords = [r'\d+点', r'几点', r'开放', r'营业', r'入住', r'退房', 
                        r'早餐.*时间', r'晚餐.*时间', r'几点']
        has_time_sensitivity = any(re.search(p, query_lower) for p in time_keywords)
        
        # 房型约束检测
        room_type_keywords = [
            r'(?:红棉|豪华|标准|套房|单人|双人|大床|双床)',
            r'(?:海景|城景|山景|湖景)',
            r'房型',
            r'(?:房间|客房).*(?:大小|面积)'
        ]
        has_room_type = any(re.search(p, query_lower) for p in room_type_keywords)
        
        # 查询长度
        query_length = len(query)
        
        # 具体性得分
        specificity_score = 0.0
        if query_length < 10:
            specificity_score = 0.2
        elif query_length < 20:
            specificity_score = 0.5
        elif query_length < 40:
            specificity_score = 0.7
        else:
            specificity_score = 0.9
        
        # 具体关键词加成
        specific_keywords = [r'\d+', r'品牌', r'型号', r'材质', r'硬度']
        for keyword in specific_keywords:
            if re.search(keyword, query_lower):
                specificity_score += 0.1
        
        specificity_score = min(1.0, specificity_score)
        
        return {
            "has_time_sensitivity": has_time_sensitivity,
            "has_room_type": has_room_type,
            "query_length": query_length,
            "specificity_score": specificity_score
        }
    
    def adjust_weights(self, base_weights: dict, query_features: dict) -> dict:
        """
        根据查询特征动态调整权重
        
        考虑因素：
        - 时间敏感性：近期查询增加评论级权重
        - 房型约束：指定房型时增加对应评论权重
        - 查询长度：长查询偏向具体，短查询偏向泛化
        
        Args:
            base_weights: 基础权重配置
            query_features: 查询特征
        
        Returns:
            调整后的权重配置
        """
        weights = base_weights.copy()
        
        # 时间敏感性调整
        if query_features["has_time_sensitivity"]:
            self._log("检测到时间敏感性，增加评论级权重")
            weights["comment"] = min(0.9, weights["comment"] + 0.2)
            weights["aspect"] = max(0.1, weights["aspect"] - 0.1)
            weights["category"] = max(0.05, weights["category"] - 0.1)
        
        # 房型约束调整
        if query_features["has_room_type"]:
            self._log("检测到房型约束，增加评论级权重")
            weights["comment"] = min(0.85, weights["comment"] + 0.15)
            weights["aspect"] = max(0.15, weights["aspect"] - 0.1)
        
        # 查询长度调整
        specificity = query_features["specificity_score"]
        if specificity > 0.7:
            # 长查询/具体查询偏向评论
            self._log(f"高具体性查询(s={specificity:.2f})，增加评论权重")
            weights["comment"] = min(0.85, weights["comment"] + 0.1 * specificity)
            weights["category"] = max(0.05, weights["category"] - 0.05 * specificity)
        elif specificity < 0.4:
            # 短查询/泛化查询偏向类别
            self._log(f"低具体性查询(s={specificity:.2f})，增加类别权重")
            weights["category"] = min(0.75, weights["category"] + 0.1 * (1 - specificity))
            weights["comment"] = max(0.1, weights["comment"] - 0.05 * (1 - specificity))
        
        # 归一化权重
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _fuse_results(
        self,
        category_results: list[dict],
        aspect_results: list[dict],
        comment_results: list[dict],
        weights: dict,
        topk: int
    ) -> list[dict]:
        """
        加权融合多粒度检索结果
        
        Args:
            category_results: 类别级检索结果
            aspect_results: 方面级检索结果
            comment_results: 评论级检索结果
            weights: 权重配置
            topk: 返回数量
        
        Returns:
            融合后的结果列表
        """
        # 为每个结果添加来源权重
        for r in category_results:
            r["_source_weight"] = weights.get("category", 0.33)
            r["_source_level"] = "category"
        for r in aspect_results:
            r["_source_weight"] = weights.get("aspect", 0.33)
            r["_source_level"] = "aspect"
        for r in comment_results:
            r["_source_weight"] = weights.get("comment", 0.33)
            r["_source_level"] = "comment"
        
        # 合并所有结果
        all_results = category_results + aspect_results + comment_results
        
        # 计算综合得分
        for r in all_results:
            base_score = r.get("score", r.get("relevance_score", 0.5))
            r["_fused_score"] = base_score * r["_source_weight"]
        
        # 按融合得分排序
        all_results.sort(key=lambda x: x["_fused_score"], reverse=True)
        
        # 去重（根据content或text字段）
        seen = set()
        unique_results = []
        for r in all_results:
            content = r.get("content", r.get("text", ""))
            content_hash = hash(content[:100] if content else "")
            if content_hash not in seen:
                seen.add(content_hash)
                # 清理内部字段
                clean_r = {k: v for k, v in r.items() if not k.startswith("_")}
                clean_r["score"] = r["_fused_score"]
                clean_r["source_level"] = r["_source_level"]
                unique_results.append(clean_r)
        
        return unique_results[:topk]
    
    def _log(self, message: str):
        """记录检索日志"""
        self.retrieval_log.append(message)
    
    def get_retrieval_log(self) -> list[str]:
        """获取检索日志"""
        return self.retrieval_log.copy()