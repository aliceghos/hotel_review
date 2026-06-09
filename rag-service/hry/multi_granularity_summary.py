"""
多粒度摘要生成模块

提供三级摘要生成功能：
- 类别级摘要（14个）
- 观点级摘要（每个类别3-5个核心观点）
- 评论级摘要（2171条，每条1句话亮点）
"""

import json
import time
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============== Prompt 模板 ==============

ASPECT_GENERATION_PROMPT = """
你是一个酒店评论分析专家。请根据以下{类别名称}的评论，分析并生成3-5个核心观点。

## 评论内容
{该类别下的所有评论文本}

## 输出要求
1. 每个观点用JSON对象表示，包含：
   - aspect_id: 观点ID，格式 ASP_{类别缩写}_{序号}
   - aspect_text: 核心观点一句话描述（30字以内）
   - aspect_detail: 观点详细摘要（100-200字）
   - keywords: 3-8个相关关键词
   - comment_count: 涉及该观点的评论数量
   - sentiment: 整体情感倾向(positive/neutral/negative/mixed)

2. 观点应具有代表性和区分度，覆盖该类别的主要方面

## 输出格式
仅输出JSON数组，不要包含其他文字。
"""

COMMENT_SUMMARY_PROMPT = """
你是一个酒店评论摘要专家。请为以下评论生成一句话摘要。

## 原始评论
{评论文本}

## 要求
1. 摘要控制在25字以内
2. 突出评论的核心亮点或关键问题
3. 保持客观准确的情感倾向

## 输出格式
仅输出JSON对象：
{
  "comment_id": "{原始ID}",
  "one_sentence_summary": "摘要内容",
  "sentiment": "positive/neutral/negative"
}
"""

CATEGORY_SUMMARY_PROMPT = """
你是一个酒店评论分析专家。请根据以下{类别名称}的评论，生成一个综合性的类别级摘要。

## 评论内容
{该类别下的所有评论文本}

## 输出要求
1. 摘要应包含：
   - summary: 类别整体概述（100-150字）
   - keywords: 5-10个核心关键词
   - sentiment_distribution: 情感分布统计 {positive/neutral/negative: 数量}

2. 突出该类别的主要特点和用户反馈

## 输出格式
仅输出JSON对象，不要包含其他文字。
"""


# ============== 数据结构 ==============

@dataclass
class CategorySummary:
    """类别级摘要"""
    category: str
    summary: str
    comment_count: int
    keywords: List[str]
    sentiment_distribution: Dict[str, int]


@dataclass
class AspectSummary:
    """观点级摘要"""
    aspect_id: str
    aspect_text: str
    aspect_detail: str
    keywords: List[str]
    comment_count: int
    sentiment: str


@dataclass
class CommentSummary:
    """评论级摘要"""
    comment_id: str
    one_sentence_summary: str
    aspect_id: str
    sentiment: str


# ============== 辅助函数 ==============

def _normalize_text(text: str) -> str:
    """标准化文本"""
    return re.sub(r"\s+", " ", text or "").strip()


def _get_category_abbreviation(category: str) -> str:
    """获取类别缩写用于生成aspect_id"""
    abbreviations = {
        "服务": "SVC",
        "设施": "FAC",
        "早餐": "BRK",
        "房间": "ROM",
        "隔音": "SND",
        "卫生": "HYG",
        "位置": "LOC",
        "停车": "PRK",
        "环境": "ENV",
        "价格": "PRC",
        "餐饮": "DIN",
        "交通": "TRF",
        "安全": "SFT",
        "体验": "EXP",
    }
    return abbreviations.get(category, category[:3].upper())


def _retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """带退避的重试装饰器"""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapper


def _parse_json_response(response: str) -> Any:
    """解析LLM返回的JSON响应"""
    response = response.strip()
    
    # 尝试提取JSON数组或对象
    json_patterns = [
        r'\[[\s\S]*\]',  # JSON数组
        r'\{[\s\S]*\}',  # JSON对象
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    
    raise ValueError(f"无法解析JSON响应: {response[:200]}")


# ============== 主类实现 ==============

class MultiGranularitySummaryGenerator:
    """多粒度摘要生成器"""
    
    def __init__(self, llm_client, config: dict = None):
        """
        初始化生成器
        
        Args:
            llm_client: LLM客户端，需支持 chat(messages) -> str 方法
            config: 配置字典，包含batch_size, max_retries等
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # 默认配置
        self.batch_size = self.config.get("batch_size", 50)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        # 生成结果存储
        self.category_summaries: Dict[str, CategorySummary] = {}
        self.aspect_summaries: Dict[str, List[AspectSummary]] = {}
        self.comment_summaries: Dict[str, CommentSummary] = {}
    
    @_retry_with_backoff
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """调用LLM接口，带重试逻辑"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.llm_client.chat(messages)
        return response
    
    def generate_category_summary(self, category: str, comments: list[dict]) -> CategorySummary:
        """
        生成类别级摘要
        
        Args:
            category: 类别名称
            comments: 该类别下的所有评论列表
        
        Returns:
            CategorySummary对象
        """
        print(f"  生成类别级摘要: {category} ({len(comments)}条评论)")
        
        # 合并所有评论文本
        all_comments_text = "\n".join([
            f"- {c.get('comment', '')}" for c in comments if c.get('comment')
        ])
        
        # 构建prompt
        prompt = CATEGORY_SUMMARY_PROMPT.format(
            类别名称=category,
            该类别下的所有评论文本=all_comments_text[:8000]  # 限制长度
        )
        
        # 调用LLM
        response = self._call_llm(prompt)
        
        # 解析响应
        try:
            data = _parse_json_response(response)
            
            # 确保情感分布存在
            sentiment_dist = data.get("sentiment_distribution", {})
            if isinstance(sentiment_dist, dict):
                for key in ["positive", "neutral", "negative", "mixed"]:
                    if key not in sentiment_dist:
                        sentiment_dist[key] = 0
            
            summary = CategorySummary(
                category=category,
                summary=data.get("summary", ""),
                comment_count=len(comments),
                keywords=data.get("keywords", []),
                sentiment_distribution=sentiment_dist
            )
            
            self.category_summaries[category] = summary
            return summary
            
        except Exception as e:
            print(f"    警告: 解析类别摘要失败 ({category}): {e}")
            # 返回默认结构
            summary = CategorySummary(
                category=category,
                summary=f"关于{category}的综合反馈，共{len(comments)}条评论。",
                comment_count=len(comments),
                keywords=[],
                sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
            )
            self.category_summaries[category] = summary
            return summary
    
    def generate_aspect_summaries(
        self, 
        category: str, 
        comments: list[dict], 
        num_aspects: int = 4
    ) -> List[AspectSummary]:
        """
        生成观点级摘要（每个类别3-5个）
        
        Args:
            category: 类别名称
            comments: 该类别下的所有评论列表
            num_aspects: 要生成的观点数量，默认4个
        
        Returns:
            AspectSummary对象列表
        """
        print(f"  生成观点级摘要: {category} ({len(comments)}条评论)")
        
        # 合并所有评论文本
        all_comments_text = "\n".join([
            f"- {c.get('comment', '')}" for c in comments if c.get('comment')
        ])
        
        # 获取类别缩写
        cat_abbrev = _get_category_abbreviation(category)
        
        # 构建prompt
        prompt = ASPECT_GENERATION_PROMPT.format(
            类别名称=category,
            该类别下的所有评论文本=all_comments_text[:10000]  # 限制长度
        )
        
        # 调用LLM
        response = self._call_llm(prompt)
        
        # 解析响应
        aspects = []
        try:
            data = _parse_json_response(response)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    aspect = AspectSummary(
                        aspect_id=item.get("aspect_id", f"ASP_{cat_abbrev}_{i+1:02d}"),
                        aspect_text=item.get("aspect_text", "")[:30],
                        aspect_detail=item.get("aspect_detail", ""),
                        keywords=item.get("keywords", [])[:8],
                        comment_count=item.get("comment_count", 0),
                        sentiment=item.get("sentiment", "neutral")
                    )
                    aspects.append(aspect)
            else:
                raise ValueError("期望返回数组格式")
                
        except Exception as e:
            print(f"    警告: 解析观点摘要失败 ({category}): {e}")
            # 返回默认结构
            for i in range(min(num_aspects, 3)):
                aspect = AspectSummary(
                    aspect_id=f"ASP_{cat_abbrev}_{i+1:02d}",
                    aspect_text=f"{category}方面的第{i+1}个观点",
                    aspect_detail="",
                    keywords=[],
                    comment_count=len(comments),
                    sentiment="neutral"
                )
                aspects.append(aspect)
        
        self.aspect_summaries[category] = aspects
        return aspects
    
    def generate_comment_summary(self, comment: dict) -> CommentSummary:
        """
        生成评论级摘要（单条评论一句话亮点）
        
        Args:
            comment: 评论字典，需包含id和comment字段
        
        Returns:
            CommentSummary对象
        """
        comment_id = comment.get("id", "")
        comment_text = comment.get("comment", "")
        
        if not comment_text:
            return CommentSummary(
                comment_id=comment_id,
                one_sentence_summary="",
                aspect_id="",
                sentiment="neutral"
            )
        
        # 构建prompt
        prompt = COMMENT_SUMMARY_PROMPT.format(
            评论文本=comment_text,
            原始ID=comment_id
        )
        
        # 调用LLM
        response = self._call_llm(prompt)
        
        # 解析响应
        try:
            data = _parse_json_response(response)
            
            summary = CommentSummary(
                comment_id=data.get("comment_id", comment_id),
                one_sentence_summary=data.get("one_sentence_summary", "")[:25],
                aspect_id=data.get("aspect_id", ""),
                sentiment=data.get("sentiment", "neutral")
            )
            
            self.comment_summaries[comment_id] = summary
            return summary
            
        except Exception as e:
            # 返回默认结构
            summary = CommentSummary(
                comment_id=comment_id,
                one_sentence_summary=comment_text[:25],
                aspect_id="",
                sentiment="neutral"
            )
            self.comment_summaries[comment_id] = summary
            return summary
    
    def generate_batch_comment_summaries(
        self, 
        comments: list[dict], 
        show_progress: bool = True
    ) -> List[CommentSummary]:
        """
        批量生成评论级摘要
        
        Args:
            comments: 评论列表
            show_progress: 是否显示进度
        
        Returns:
            CommentSummary列表
        """
        results = []
        total = len(comments)
        
        for i, comment in enumerate(comments):
            if show_progress and (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{total} ({100*(i+1)//total}%)")
            
            summary = self.generate_comment_summary(comment)
            results.append(summary)
        
        if show_progress:
            print(f"  完成: {total}/{total} (100%)")
        
        return results
    
    def generate_all_summaries(
        self, 
        comments_by_category: dict,
        show_progress: bool = True
    ) -> dict:
        """
        为所有类别生成完整的多粒度摘要
        
        Args:
            comments_by_category: 按类别分组的评论字典 {category: [comments]}
            show_progress: 是否显示进度
        
        Returns:
            包含所有级别摘要的字典
        """
        categories = list(comments_by_category.keys())
        total_categories = len(categories)
        
        print(f"\n{'='*60}")
        print(f"开始生成多粒度摘要")
        print(f"类别数量: {total_categories}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 1. 生成类别级和观点级摘要
        for idx, category in enumerate(categories, 1):
            if show_progress:
                print(f"\n[类别 {idx}/{total_categories}] {category}")
            
            comments = comments_by_category[category]
            
            # 生成类别级摘要
            self.generate_category_summary(category, comments)
            
            # 生成观点级摘要
            self.generate_aspect_summaries(category, comments)
            
            if show_progress:
                print(f"  已完成")
        
        # 2. 收集所有评论用于生成评论级摘要
        all_comments = []
        for category, comments in comments_by_category.items():
            all_comments.extend(comments)
        
        print(f"\n[评论级摘要] 共 {len(all_comments)} 条评论")
        
        # 3. 批量生成评论级摘要
        self.generate_batch_comment_summaries(all_comments, show_progress=show_progress)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"多粒度摘要生成完成!")
        print(f"总耗时: {elapsed:.2f}秒")
        print(f"类别摘要: {len(self.category_summaries)} 个")
        print(f"观点摘要: {sum(len(v) for v in self.aspect_summaries.values())} 个")
        print(f"评论摘要: {len(self.comment_summaries)} 条")
        print(f"{'='*60}\n")
        
        return {
            "category_summaries": self.category_summaries,
            "aspect_summaries": self.aspect_summaries,
            "comment_summaries": self.comment_summaries
        }
    
    def save_summaries(self, summaries: dict, output_dir: str):
        """
        保存摘要到文件
        
        Args:
            summaries: generate_all_summaries返回的摘要字典
            output_dir: 输出目录路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存类别级摘要
        category_file = output_path / "category_summaries.json"
        category_data = [
            asdict(s) for s in summaries["category_summaries"].values()
        ]
        with category_file.open("w", encoding="utf-8") as f:
            json.dump(category_data, f, ensure_ascii=False, indent=2)
        print(f"已保存类别级摘要: {category_file}")
        
        # 2. 保存观点级摘要
        aspect_file = output_path / "aspect_summaries.json"
        aspect_data = {}
        for category, aspects in summaries["aspect_summaries"].items():
            aspect_data[category] = [asdict(a) for a in aspects]
        with aspect_file.open("w", encoding="utf-8") as f:
            json.dump(aspect_data, f, ensure_ascii=False, indent=2)
        print(f"已保存观点级摘要: {aspect_file}")
        
        # 3. 保存评论级摘要
        comment_file = output_path / "comment_summaries.json"
        comment_data = [
            asdict(s) for s in summaries["comment_summaries"].values()
        ]
        with comment_file.open("w", encoding="utf-8") as f:
            json.dump(comment_data, f, ensure_ascii=False, indent=2)
        print(f"已保存评论级摘要: {comment_file}")
        
        # 4. 保存汇总统计
        stats_file = output_path / "summary_stats.json"
        stats = {
            "generated_at": datetime.now().isoformat(),
            "category_count": len(summaries["category_summaries"]),
            "aspect_count": sum(len(v) for v in summaries["aspect_summaries"].values()),
            "comment_count": len(summaries["comment_summaries"]),
            "categories": list(summaries["category_summaries"].keys()),
        }
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存统计信息: {stats_file}")


# ============== 使用示例 ==============

def demo_usage():
    """演示如何使用多粒度摘要生成器"""
    
    # 模拟LLM客户端
    class MockLLMClient:
        def chat(self, messages):
            # 模拟返回JSON
            return '[{"aspect_id": "ASP_SVC_01", "aspect_text": "前台服务热情专业", "aspect_detail": "前台工作人员态度好", "keywords": ["前台", "服务", "热情"], "comment_count": 50, "sentiment": "positive"}]'
    
    # 创建生成器
    llm_client = MockLLMClient()
    generator = MultiGranularitySummaryGenerator(llm_client)
    
    # 示例数据
    sample_comments = {
        "服务": [
            {"id": "c1", "comment": "前台服务非常热情，办理入住很快。"},
            {"id": "c2", "comment": "服务人员态度很好，房间干净整洁。"},
        ],
        "设施": [
            {"id": "c3", "comment": "房间设施齐全，床很舒服。"},
        ]
    }
    
    # 生成摘要
    summaries = generator.generate_all_summaries(sample_comments)
    
    # 保存
    generator.save_summaries(summaries, "output/summaries")
    
    return summaries


if __name__ == "__main__":
    demo_usage()
