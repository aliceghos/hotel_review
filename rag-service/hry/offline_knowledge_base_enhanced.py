"""
增强版离线知识库构建器

支持 LLM mode / doc2query mode / hybrid mode 三种反向Query生成模式切换。
"""

import json
import time
import logging
import pickle
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryGenerationResult:
    """单个评论的反向Query生成结果"""
    comment_id: str
    generated_queries: List[str]
    generation_mode: str  # "llm" | "doc2query" | "hybrid"
    generation_time_ms: int
    success: bool = True
    error_message: str = ""


class CheckpointManager:
    """检查点管理器，支持断点续传"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """加载检查点"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"从检查点恢复: {len(data.get('completed_ids', []))} 条已完成")
                return data
        return {"completed_ids": [], "results": [], "metadata": {}}

    def save(self, completed_ids: List[str], results: List[Dict], metadata: Dict):
        """保存检查点"""
        checkpoint_data = {
            "completed_ids": completed_ids,
            "results": results,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"检查点已保存: {self.checkpoint_path}")


class OfflineKnowledgeBaseBuilder:
    """
    增强版离线知识库构建器

    支持三种反向Query生成模式：
    - llm: 使用LLM生成
    - doc2query: 使用doc2query模型生成
    - hybrid: 两者结合
    """

    # LLM生成提示词模板
    LLM_PROMPT_TEMPLATE = """你是一个酒店评论问答助手。根据以下评论内容，生成3个用户可能搜索的问题。

评论内容：{comment}

要求：
1. 问题应该简洁、具体
2. 覆盖评论的不同方面
3. 使用自然语言，符合用户查询习惯

生成的问题："""

    # LLM优化提示词模板（用于混合模式）
    LLM_OPTIMIZE_TEMPLATE = """你是一个酒店评论问答助手。请优化以下候选查询，使其更符合用户搜索习惯。

原始评论：{comment}

候选查询：
{queries}

请：
1. 筛选出最相关的1-3个查询
2. 优化表达方式，使其更自然
3. 保持查询的多样性

优化后的查询："""

    # 重试配置
    DEFAULT_RETRY_CONFIG = {
        "max_retries": 3,
        "retry_delay": 1.0,
        "backoff_factor": 2.0
    }

    def __init__(
        self,
        config: dict,
        llm_client=None,
        doc2query_model=None
    ):
        """
        初始化增强版离线知识库构建器

        Args:
            config: 配置字典，包含：
                - query_generation_mode: "llm" | "doc2query" | "hybrid"
                - doc2query_model_path: doc2query模型路径
                - checkpoint_path: 检查点文件路径
                - output_path: 输出文件路径
                - llm_model_name: LLM模型名称（可选）
            llm_client: LLM客户端对象（可选）
            doc2query_model: doc2query模型对象（可选）
        """
        self.config = config
        self.query_generation_mode = config.get("query_generation_mode", "doc2query")
        self.doc2query_model_path = config.get("doc2query_model_path")
        self.checkpoint_path = config.get("checkpoint_path", "data/query_generation_checkpoint.json")
        self.output_path = config.get("output_path", "data/reverse_queries_enhanced.json")

        self.llm_client = llm_client
        self.doc2query_model = doc2query_model

        self.checkpoint_manager = CheckpointManager(self.checkpoint_path)

        # 初始化doc2query模型
        if self.doc2query_model is None and self.query_generation_mode in ["doc2query", "hybrid"]:
            self._init_doc2query_model()

        logger.info(f"OfflineKnowledgeBaseBuilder 初始化完成，模式: {self.query_generation_mode}")

    def _init_doc2query_model(self):
        """初始化doc2query模型"""
        try:
            from doc2query_model import Doc2QueryModel, GPT2BasedGenerator
            self.doc2query_model = GPT2BasedGenerator(
                model_path=self.doc2query_model_path,
                device="cuda" if __import__('torch').cuda.is_available() else "cpu"
            )
            logger.info("doc2query模型加载成功")
        except Exception as e:
            logger.warning(f"doc2query模型加载失败: {e}，将使用备用模式")
            self.doc2query_model = None

    def generate_reverse_queries(
        self,
        comments: List[dict],
        mode: str = None,
        use_checkpoint: bool = True,
        show_progress: bool = True
    ) -> List[QueryGenerationResult]:
        """
        生成反向Query

        Args:
            comments: 评论列表，每个字典包含 id, comment 等字段
            mode: 生成模式，"llm" | "doc2query" | "hybrid"，默认使用初始化时的模式
            use_checkpoint: 是否使用断点续传
            show_progress: 是否显示进度

        Returns:
            生成结果列表
        """
        mode = mode or self.query_generation_mode
        logger.info(f"开始生成反向Query，模式: {mode}，评论数量: {len(comments)}")

        # 加载检查点
        completed_ids = []
        existing_results = []
        if use_checkpoint:
            checkpoint_data = self.checkpoint_manager.load()
            completed_ids = checkpoint_data.get("completed_ids", [])
            existing_results = checkpoint_data.get("results", [])
            logger.info(f"从检查点恢复，已完成 {len(completed_ids)} 条")

        # 过滤未完成的评论
        remaining_comments = [c for c in comments if str(c.get("id", c.get("comment_id", ""))) not in completed_ids]

        if not remaining_comments:
            logger.info("所有评论已处理完成")
            return [QueryGenerationResult(**r) for r in existing_results]

        logger.info(f"还需处理 {len(remaining_comments)} 条评论")

        # 生成结果
        results = existing_results.copy()
        total = len(comments)

        for i, comment in enumerate(remaining_comments):
            comment_id = str(comment.get("id", comment.get("comment_id", f"comment_{i}")))
            comment_text = comment.get("comment", comment.get("text", ""))

            if show_progress:
                processed = len(completed_ids) + i + 1
                logger.info(f"[{processed}/{total}] 处理评论: {comment_id}")

            try:
                result = self._generate_single_comment_query(comment_text, comment_id, mode)
                results.append(asdict(result))
            except Exception as e:
                logger.error(f"处理评论 {comment_id} 失败: {e}")
                results.append(asdict(QueryGenerationResult(
                    comment_id=comment_id,
                    generated_queries=[],
                    generation_mode=mode,
                    generation_time_ms=0,
                    success=False,
                    error_message=str(e)
                )))

            # 保存检查点
            if use_checkpoint and (i + 1) % 10 == 0:
                completed_ids_current = completed_ids + [c.get("id", c.get("comment_id", "")) for c in remaining_comments[:i+1]]
                self.checkpoint_manager.save(
                    completed_ids_current,
                    results,
                    {"mode": mode, "total": total}
                )

        # 最终保存
        if use_checkpoint:
            all_completed_ids = completed_ids + [str(c.get("id", c.get("comment_id", ""))) for c in remaining_comments]
            self.checkpoint_manager.save(all_completed_ids, results, {"mode": mode, "total": total})

        return [QueryGenerationResult(**r) for r in results]

    def _generate_single_comment_query(
        self,
        comment_text: str,
        comment_id: str,
        mode: str
    ) -> QueryGenerationResult:
        """为单条评论生成反向Query"""
        start_time = time.time()

        if mode == "llm":
            queries = self._llm_query_generation(comment_text)
        elif mode == "doc2query":
            queries = self._doc2query_generation(comment_text)
        elif mode == "hybrid":
            queries = self._hybrid_query_generation(comment_text)
        else:
            raise ValueError(f"不支持的生成模式: {mode}")

        generation_time_ms = int((time.time() - start_time) * 1000)

        return QueryGenerationResult(
            comment_id=comment_id,
            generated_queries=queries,
            generation_mode=mode,
            generation_time_ms=generation_time_ms,
            success=True
        )

    def _llm_query_generation(self, comment: str, num_queries: int = 3) -> List[str]:
        """
        使用LLM生成反向Query

        Args:
            comment: 评论文本
            num_queries: 生成数量

        Returns:
            生成的查询列表
        """
        if self.llm_client is None:
            logger.warning("LLM客户端未设置，使用默认查询")
            return self._generate_fallback_queries(comment, num_queries)

        prompt = self.LLM_PROMPT_TEMPLATE.format(comment=comment)

        for retry in range(self.DEFAULT_RETRY_CONFIG["max_retries"]):
            try:
                response = self._call_llm(prompt)
                queries = self._parse_llm_response(response, num_queries)
                if queries:
                    return queries
            except Exception as e:
                logger.warning(f"LLM生成失败 (重试 {retry + 1}): {e}")
                time.sleep(self.DEFAULT_RETRY_CONFIG["retry_delay"] * (self.DEFAULT_RETRY_CONFIG["backoff_factor"] ** retry))

        return self._generate_fallback_queries(comment, num_queries)

    def _doc2query_generation(self, comment: str, num_queries: int = 3) -> List[str]:
        """
        使用doc2query模型生成反向Query

        Args:
            comment: 评论文本
            num_queries: 生成数量

        Returns:
            生成的查询列表
        """
        if self.doc2query_model is None:
            logger.warning("doc2query模型未加载，使用默认查询")
            return self._generate_fallback_queries(comment, num_queries)

        try:
            queries = self.doc2query_model.generate(comment, num_queries)
            return queries if queries else self._generate_fallback_queries(comment, num_queries)
        except Exception as e:
            logger.error(f"doc2query模型推理失败: {e}")
            return self._generate_fallback_queries(comment, num_queries)

    def _hybrid_query_generation(self, comment: str, num_queries: int = 3) -> List[str]:
        """
        混合模式生成反向Query

        流程：
        1. 使用doc2query快速生成多个候选query
        2. 使用LLM对候选进行优化和筛选
        3. 确保生成1-3个高质量query

        Args:
            comment: 评论文本
            num_queries: 生成数量

        Returns:
            优化后的查询列表
        """
        # 步骤1: doc2query快速生成候选
        candidate_queries = self._doc2query_generation(comment, num_queries * 2)

        if not candidate_queries:
            candidate_queries = self._generate_fallback_queries(comment, num_queries)

        # 步骤2: 如果有LLM，使用LLM优化
        if self.llm_client is not None:
            try:
                optimized_queries = self._llm_optimize_queries(comment, candidate_queries, num_queries)
                return optimized_queries
            except Exception as e:
                logger.warning(f"LLM优化失败，回退到候选查询: {e}")

        # 步骤3: 返回候选（去重后限制数量）
        unique_queries = list(dict.fromkeys(candidate_queries))
        return unique_queries[:num_queries]

    def _llm_optimize_queries(
        self,
        comment: str,
        candidate_queries: List[str],
        num_queries: int = 3
    ) -> List[str]:
        """
        使用LLM优化和筛选候选查询

        Args:
            comment: 原始评论
            candidate_queries: 候选查询列表
            num_queries: 目标数量

        Returns:
            优化后的查询列表
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(candidate_queries)])
        prompt = self.LLM_OPTIMIZE_TEMPLATE.format(
            comment=comment[:200],
            queries=queries_text
        )

        for retry in range(self.DEFAULT_RETRY_CONFIG["max_retries"]):
            try:
                response = self._call_llm(prompt)
                queries = self._parse_llm_response(response, num_queries)
                if queries:
                    return queries
            except Exception as e:
                logger.warning(f"LLM优化失败 (重试 {retry + 1}): {e}")
                time.sleep(self.DEFAULT_RETRY_CONFIG["retry_delay"] * (self.DEFAULT_RETRY_CONFIG["backoff_factor"] ** retry))

        # 回退到候选查询
        unique_queries = list(dict.fromkeys(candidate_queries))
        return unique_queries[:num_queries]

    def _call_llm(self, prompt: str) -> str:
        """调用LLM接口"""
        if self.llm_client is None:
            raise RuntimeError("LLM客户端未设置")

        # 通用LLM调用接口
        if hasattr(self.llm_client, "generate"):
            return self.llm_client.generate(prompt)
        elif hasattr(self.llm_client, "call"):
            return self.llm_client.call(prompt)
        elif hasattr(self.llm_client, "__call__"):
            return self.llm_client(prompt)
        else:
            raise RuntimeError(f"LLM客户端不支持的接口类型: {type(self.llm_client)}")

    def _parse_llm_response(self, response: str, num_queries: int = 3) -> List[str]:
        """解析LLM响应，提取查询列表"""
        queries = []

        # 按行分割
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            # 匹配常见格式: "1. 查询内容", "- 查询内容", "查询内容"
            match = re.match(r'^[\d\-\*\•]+\.?\s*(.+?)[？?。.]?$', line)
            if match:
                query = match.group(1).strip()
                if query and len(query) >= 4:
                    queries.append(query)
            elif "？" in line or "?" in line:
                # 直接是问句
                query = line.strip().rstrip("。.")
                if query and len(query) >= 4:
                    queries.append(query)

        # 去重
        unique_queries = list(dict.fromkeys(queries))
        return unique_queries[:num_queries]

    def _generate_fallback_queries(self, comment: str, num_queries: int = 3) -> List[str]:
        """生成备用查询（当模型不可用时）"""
        # 从评论中提取关键词
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', comment)
        entities = list(dict.fromkeys(words))[:3]

        queries = []
        for entity in entities[:num_queries]:
            queries.append(f"关于\"{entity}\"有什么信息?")

        if len(queries) < num_queries:
            queries.append(f"这家酒店怎么样?")

        return queries[:num_queries]

    def save_results(
        self,
        results: List[QueryGenerationResult],
        output_path: str = None
    ):
        """
        保存生成结果

        Args:
            results: 生成结果列表
            output_path: 输出文件路径
        """
        output_path = output_path or self.output_path

        output_data = {
            "metadata": {
                "query_generation_mode": self.query_generation_mode,
                "total_count": len(results),
                "success_count": sum(1 for r in results if r.success),
                "timestamp": datetime.now().isoformat()
            },
            "results": [asdict(r) for r in results]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {output_path}")

    def export_for_knowledge_base(
        self,
        results: List[QueryGenerationResult],
        comments_df=None,
        output_path: str = None
    ) -> List[dict]:
        """
        导出为知识库兼容格式

        Args:
            results: 生成结果
            comments_df: 原始评论DataFrame（可选）
            output_path: 输出文件路径

        Returns:
            知识库格式的数据列表
        """
        knowledge_base_data = []

        for result in results:
            if not result.success or not result.generated_queries:
                continue

            # 获取原始评论信息
            comment_info = {}
            if comments_df is not None:
                try:
                    row = comments_df.loc[result.comment_id]
                    comment_info = {
                        "comment": row.get("comment", ""),
                        "room_type": row.get("room_type", ""),
                        "fuzzy_room_type": row.get("fuzzy_room_type", "")
                    }
                except KeyError:
                    pass

            for query in result.generated_queries:
                knowledge_base_data.append({
                    "query": query,
                    "comment_id": result.comment_id,
                    "generation_mode": result.generation_mode,
                    "generation_time_ms": result.generation_time_ms,
                    **comment_info
                })

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df = comments_df.__class__(knowledge_base_data) if comments_df is not None else None
            if df is not None:
                df.to_csv(output_path)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(knowledge_base_data, f, ensure_ascii=False, indent=2)

        logger.info(f"导出知识库数据 {len(knowledge_base_data)} 条")
        return knowledge_base_data

    def get_statistics(self, results: List[QueryGenerationResult]) -> Dict[str, Any]:
        """
        获取生成结果统计信息

        Args:
            results: 生成结果列表

        Returns:
            统计信息字典
        """
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success

        mode_counts = {}
        for r in results:
            mode_counts[r.generation_mode] = mode_counts.get(r.generation_mode, 0) + 1

        time_ms = [r.generation_time_ms for r in results if r.success]
        avg_time_ms = sum(time_ms) / len(time_ms) if time_ms else 0

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "mode_counts": mode_counts,
            "avg_generation_time_ms": avg_time_ms,
            "total_generation_time_s": sum(time_ms) / 1000 if time_ms else 0
        }


class MockLLMClient:
    """模拟LLM客户端，用于测试"""

    def __init__(self, response_template: str = None):
        self.response_template = response_template or "这家酒店怎么样？服务好吗？价格如何？"

    def generate(self, prompt: str) -> str:
        return self.response_template

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


def demo():
    """演示用法"""
    # 示例评论
    sample_comments = [
        {"id": "c001", "comment": "酒店位置很好，距离市中心很近，交通便利。房间干净整洁，服务人员态度热情。"},
        {"id": "c002", "comment": "早餐种类丰富，味道不错。床铺舒适，睡眠质量高。"},
        {"id": "c003", "comment": "价格偏高，但性价比还不错。设施齐全，推荐入住。"},
    ]

    print("=" * 60)
    print("增强版离线知识库构建器演示")
    print("=" * 60)

    # 配置
    config = {
        "query_generation_mode": "doc2query",
        "doc2query_model_path": None,  # 使用默认模型
        "checkpoint_path": "data/demo_checkpoint.json",
        "output_path": "data/demo_reverse_queries.json"
    }

    # 初始化构建器
    builder = OfflineKnowledgeBaseBuilder(config)

    # 生成反向Query（使用doc2query模式）
    print("\n【doc2query 模式】")
    results = builder.generate_reverse_queries(sample_comments, mode="doc2query")
    for r in results:
        print(f"评论 {r.comment_id}: {r.generated_queries}")

    # 统计信息
    stats = builder.get_statistics(results)
    print(f"\n统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    # 保存结果
    builder.save_results(results)

    # 导出知识库格式
    knowledge_base_data = builder.export_for_knowledge_base(results)
    print(f"\n导出知识库数据: {len(knowledge_base_data)} 条")

    # 使用混合模式
    print("\n【混合模式】")
    mock_llm = MockLLMClient()
    builder_hybrid = OfflineKnowledgeBaseBuilder(config, llm_client=mock_llm)
    results_hybrid = builder_hybrid.generate_reverse_queries(sample_comments, mode="hybrid")
    for r in results_hybrid:
        print(f"评论 {r.comment_id}: {r.generated_queries}")


if __name__ == "__main__":
    demo()
