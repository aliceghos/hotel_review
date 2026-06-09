"""酒店评论 RAG 系统：完整的检索增强生成工作流

合并了 CSW（ES/向量抽象/多样性重排）和 YTY（上下文解析/比较查询/历史压缩/评估）的优化。
"""

import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import chromadb

from config import TODAY, EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES
from modules.clients import LLMClient, EmbeddingClient
from modules.index import InvertedIndex
from modules.intent import (
    IntentRecognizer, IntentDetector, IntentExpander,
    HyDEGenerator, QueryContextResolver
)
from modules.retriever import HybridRetriever
from modules.ranker import Reranker, MultiFactorRanker
from modules.generator import ResponseGenerator
from modules.history_manager import ConversationHistoryManager
from modules.vector_store import create_vector_search_engine
from modules.chunking import ChunkIndexer
from utils.database import get_all_comments_from_insforge


class HotelReviewRAG:
    """酒店评论 RAG 系统：完整的检索增强生成工作流"""

    def __init__(self, api_key: str, dashvector_api_key: str, dashvector_endpoint: str,
                 data_dir: Path, df_comments: pd.DataFrame = None,
                 intl_api_key: str = None,
                 detection_model: str = "qwen-plus",
                 expansion_hyde_model: str = "qwen-flash",
                 generation_model: str = "qwen-plus"):
        """
        初始化 RAG 系统

        参数:
            api_key: DashScope API Key（北京）
            dashvector_api_key: DashVector API Key
            dashvector_endpoint: DashVector 集合端点
            data_dir: 数据目录（包含 chroma_db/）
            df_comments: 评论 DataFrame（若为 None 则从 Insforge 数据库加载）
            intl_api_key: DashScope API Key（新加坡，可选）
            detection_model: 意图检测模型
            expansion_hyde_model: 意图扩展/HyDE 模型
            generation_model: 回复生成模型
        """
        # 连接向量数据库（CSW: 向量检索引擎抽象层）
        vector_engine = create_vector_search_engine(
            os.getenv("VECTOR_SEARCH_ENGINE", "dashvector"),
            dashvector_api_key,
            dashvector_endpoint
        )
        self.comments_collection = vector_engine.comments
        self.reverse_queries_collection = vector_engine.reverse_queries

        chroma_db_path = data_dir / "chroma_db"
        chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
        self.summaries_collection = chroma_client.get_collection("summary_database")

        # 加载评论数据
        if df_comments is not None:
            self.df_comments = df_comments
        else:
            self.df_comments = get_all_comments_from_insforge()

        # 文本索引：Elasticsearch 可用则用，否则回退到本地倒排索引（CSW 优化7）
        es_url = os.getenv("ELASTICSEARCH_URL")
        if es_url:
            from modules.text_search import ElasticsearchBM25Index
            self.text_index = ElasticsearchBM25Index.from_env()
            auto_index = os.getenv("ELASTICSEARCH_AUTO_INDEX", "true").lower() not in {
                "0", "false", "no",
            }
            self.text_index.ensure_index(self.df_comments, auto_index=auto_index)
            self._use_es = True
        else:
            self.inverted_index = InvertedIndex()
            self.inverted_index.load(str(data_dir / "inverted_index.pkl"))
            self.text_index = self.inverted_index
            self._use_es = False

        # 确定 API Key
        key = intl_api_key if intl_api_key else api_key

        # 初始化各组件
        detection_client = LLMClient(key, model=detection_model, json=True)
        expansion_hyde_client = LLMClient(key, model=expansion_hyde_model, json=True)
        embedding_client = EmbeddingClient(key)

        self.intent_recognizer = IntentRecognizer(key)
        self.intent_detector = IntentDetector(
            detection_client, EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES
        )
        self.intent_expander = IntentExpander(expansion_hyde_client)
        self.hyde_generator = HyDEGenerator(expansion_hyde_client)
        # YTY 方向9：上下文感知查询解析
        self.query_context_resolver = QueryContextResolver(expansion_hyde_client)
        # YTY 方向19深化：历史摘要压缩（独立 qwen-flash 客户端，节省成本）
        summary_client = LLMClient(key, model="qwen-flash", json=False)
        self.history_manager = ConversationHistoryManager(summary_client)
        # 优化2：评论分块索引
        self.chunk_indexer = ChunkIndexer(data_dir)
        self.chunk_indexer.build_if_empty(self.df_comments, embedding_client)
        self.retriever = HybridRetriever(
            self.text_index, self.comments_collection,
            self.reverse_queries_collection, self.summaries_collection,
            embedding_client, self.df_comments, self.hyde_generator,
            use_es=self._use_es, chunk_indexer=self.chunk_indexer
        )
        # CSW 方向15：embedding_client 保留供 DiversityReranker 使用
        self.embedding_client = embedding_client
        self.reranker = Reranker(key)
        self.generator = ResponseGenerator(key, model=generation_model)

    def query(self, user_query: str,
              route_topk: int = 150,
              retrieval_topk: int = 100,
              ranking_topk: int = 10,
              enable_expansion: bool = True,
              enable_bm25: bool = True,
              enable_vector: bool = True,
              enable_reverse: bool = True,
              enable_hyde: bool = True,
              enable_summary: bool = True,
              enable_ranking: bool = True,
              enable_generation: bool = True,
              print_response: bool = True,
              w_relevance: float = 0.40,
              w_quality: float = 0.25,
              w_length: float = 0.05,
              w_review: float = 0.05,
              w_useful: float = 0.05,
              w_recency: float = 0.20,
              base_decay: float = 0.5,
              implied_boost: float = 0.5,
              clear_boost: float = 0.5,
              half_life_days: int = 180,
              enable_diversity: bool = True,
              diversity_strategy: str = "mmr",
              diversity_lambda: float = 0.72,
              diversity_pool_size: int = 30,
              today: datetime | None = TODAY,
              history: list | dict | None = None) -> dict:
        """
        处理用户查询并生成回复

        返回:
            {
                'response': 模型回复文本,
                'evaluation': 回复质量评估（YTY方向16深化）,
                'references': { 'comments', 'summaries', 'hyde_responses' },
                'query_processing': { 'intent_recognition', 'intent_detection',
                    'intent_expansion', 'query_type', 'resolved_query' },
                'timing': { ... }
            }
        """
        total_start = time.time()
        timing = {}
        if not today:
            today = datetime.today()

        # 一、查询处理
        query_processing_start = time.time()

        # 1. 意图识别
        intent_recognition_start = time.time()
        need_retrieval = self.intent_recognizer.recognize(user_query)
        timing['intent_recognition'] = time.time() - intent_recognition_start

        # 1.2 历史摘要压缩（YTY方向19深化）
        # 当对话轮数超过阈值时，将较早的历史压缩为摘要条目，控制 token 消耗
        if history and isinstance(history, list):
            history, timing['history_compress'] = self._timed_call(
                self.history_manager.compress, history
            )
        else:
            timing['history_compress'] = 0

        # 1.5 上下文感知查询解析（YTY方向9）
        # 将含指代/省略的追问改写为完整独立查询，供后续检索使用
        resolved_query = user_query
        query_type = "normal"  # YTY方向9深化
        timing['context_resolution'] = 0
        if need_retrieval and history:
            history_list = history if isinstance(history, list) else ([history] if isinstance(history, dict) else [])
            if history_list:
                resolved_query, timing['context_resolution'] = self._timed_call(
                    self.query_context_resolver.resolve, user_query, history_list
                )
                if resolved_query != user_query:
                    print(f"[QueryContextResolver] 查询已解析\n  原始: {user_query}\n  解析后: {resolved_query}")

        # 2. 意图检测与意图扩展（使用解析后的查询）
        intent_detection_result = None
        intent_expansion_result = None
        timing['intent_detection'] = 0
        timing['intent_expansion'] = 0

        if need_retrieval:
            if enable_expansion:
                # 在并行执行前先检测 query_type（YTY方向9深化：比较型查询路由）
                query_type = "comparative" if self.intent_expander.is_comparative_query(resolved_query) else "normal"
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_detect = executor.submit(
                        self._timed_call, self.intent_detector.detect, resolved_query
                    )
                    future_expand = executor.submit(
                        self._timed_call, self.intent_expander.expand, resolved_query
                    )
                    intent_detection_result, timing['intent_detection'] = future_detect.result()
                    intent_expansion_result, timing['intent_expansion'] = future_expand.result()
            else:
                intent_detection_result, timing['intent_detection'] = self._timed_call(
                    self.intent_detector.detect, resolved_query
                )
                intent_expansion_result, timing['intent_expansion'] = None, 0

        timing['query_processing_total'] = time.time() - query_processing_start

        # 直接回答
        if not need_retrieval:
            if enable_generation:
                first_token_base = time.time() - total_start
                response, ttft_model, subsequent, generation, evaluation = self.generator.generate(
                    user_query, need_retrieval=False, print_response=print_response,
                    today=today, history=history, enable_evaluation=False
                )
                timing['ttft'] = first_token_base + ttft_model
                timing['ttft_model'] = ttft_model
                timing['subsequent'] = subsequent
                timing['generation'] = generation
            else:
                response = ""
                evaluation = {}
                timing['ttft'] = 0
                timing['ttft_model'] = 0
                timing['subsequent'] = 0
                timing['generation'] = 0

            timing['total'] = time.time() - total_start
            return {
                'response': response,
                'evaluation': evaluation,
                'references': {'comments': [], 'summaries': [], 'hyde_responses': {}},
                'query_processing': {
                    'intent_recognition': need_retrieval,
                    'intent_detection': None,
                    'intent_expansion': None,
                    'query_type': query_type,
                    'resolved_query': resolved_query
                },
                'timing': timing
            }

        # 二、混合检索
        if enable_ranking:
            final_topk_for_retrieval = retrieval_topk
        else:
            final_topk_for_retrieval = ranking_topk

        rewritten_queries = (intent_expansion_result
                             if intent_expansion_result
                             else [{'query': resolved_query, 'weight': 1.0}])

        comments, summaries, retrieval_timing, hyde_results = self.retriever.retrieve(
            rewritten_queries,
            room_type=intent_detection_result.get('room_type'),
            fuzzy_room_type=intent_detection_result.get('fuzzy_room_type'),
            topk=route_topk,
            final_topk=final_topk_for_retrieval,
            enable_bm25=enable_bm25,
            enable_vector=enable_vector,
            enable_reverse=enable_reverse,
            enable_hyde=enable_hyde,
            enable_summary=enable_summary
        )
        timing['retrieval'] = retrieval_timing

        # 三、排序
        if enable_ranking:
            # CSW方向15：MultiFactorRanker 支持 embedding_client 用于 DiversityReranker
            ranker = MultiFactorRanker(
                self.reranker,
                embedding_client=self.embedding_client,
                w_relevance=w_relevance, w_quality=w_quality,
                w_length=w_length, w_review=w_review,
                w_useful=w_useful, w_recency=w_recency,
                base_decay=base_decay, implied_boost=implied_boost,
                clear_boost=clear_boost, half_life_days=half_life_days
            )
            ranked_comments, ranking_timing = ranker.rank(
                user_query, comments,
                time_sensitivity=intent_detection_result.get('time_sensitivity'),
                topk=ranking_topk, today=today,
                enable_diversity=enable_diversity,
                diversity_strategy=diversity_strategy,
                diversity_lambda=diversity_lambda,
                diversity_pool_size=diversity_pool_size
            )
            timing['ranking'] = ranking_timing
        else:
            ranked_comments = comments
            timing['ranking'] = {'total': 0, 'rerank': 0, 'scoring': 0, 'diversity': 0}

        # 四、回复生成
        if enable_generation:
            first_token_base = time.time() - total_start
            response, ttft_model, subsequent, generation, evaluation = self.generator.generate(
                user_query,
                rewritten_queries=intent_expansion_result,
                ranked_comments=ranked_comments,
                summaries=summaries,
                need_retrieval=True,
                print_response=print_response,
                today=today,
                history=history
            )
            timing['ttft'] = first_token_base + ttft_model
            timing['ttft_model'] = ttft_model
            timing['subsequent'] = subsequent
            timing['generation'] = generation
        else:
            response = ""
            evaluation = {}
            timing['ttft'] = 0
            timing['ttft_model'] = 0
            timing['subsequent'] = 0
            timing['generation'] = 0

        timing['total'] = time.time() - total_start

        # 五、构建返回结果
        processed_comments = []
        for c in ranked_comments:
            comment_data = {
                'comment_id': c['comment_id'],
                'comment': c['comment'],
                'rrf_score': c['rrf_score'],
                'rrf_rank': c['rrf_rank'],
                'route_ranks': c['route_ranks'],
                'metadata': c['metadata']
            }
            if enable_ranking:
                comment_data['rerank_score'] = c['rerank_score']
                comment_data['rerank_rank'] = c['rerank_rank']
                comment_data['final_score'] = c['final_score']
                comment_data['final_rank'] = c['final_rank']
                comment_data['pre_diversity_rank'] = c.get('pre_diversity_rank')
                comment_data['diversity_strategy'] = c.get('diversity_strategy')
                comment_data['feature_scores'] = c['feature_scores']
            processed_comments.append(comment_data)

        return {
            'response': response,
            'evaluation': evaluation,
            'references': {
                'comments': processed_comments,
                'summaries': summaries,
                'hyde_responses': hyde_results
            },
            'query_processing': {
                'intent_recognition': need_retrieval,
                'intent_detection': intent_detection_result,
                'intent_expansion': intent_expansion_result,
                'query_type': query_type,
                'resolved_query': resolved_query
            },
            'timing': timing
        }

    def query_stream(self, user_query: str,
                     route_topk: int = 150,
                     retrieval_topk: int = 100,
                     ranking_topk: int = 10,
                     enable_expansion: bool = True,
                     enable_bm25: bool = True,
                     enable_vector: bool = True,
                     enable_reverse: bool = True,
                     enable_hyde: bool = False,
                     enable_summary: bool = True,
                     enable_ranking: bool = True,
                     w_relevance: float = 0.40,
                     w_quality: float = 0.25,
                     w_length: float = 0.05,
                     w_review: float = 0.05,
                     w_useful: float = 0.05,
                     w_recency: float = 0.20,
                     base_decay: float = 0.5,
                     implied_boost: float = 0.5,
                     clear_boost: float = 0.5,
                     half_life_days: int = 180,
                     enable_diversity: bool = True,
                     diversity_strategy: str = "mmr",
                     diversity_lambda: float = 0.72,
                     diversity_pool_size: int = 30,
                     today: datetime | None = TODAY,
                     history: list | dict | None = None):
        """
        流式处理用户查询（用于 FastAPI SSE 接口）

        Yields:
            dict: SSE 事件
        """
        total_start = time.time()
        timing = {}
        if not today:
            today = datetime.today()

        # 一、查询处理
        query_processing_start = time.time()

        intent_recognition_start = time.time()
        need_retrieval = self.intent_recognizer.recognize(user_query)
        timing['intent_recognition'] = time.time() - intent_recognition_start

        yield {"type": "intent", "data": {"need_retrieval": need_retrieval}}

        # 1.2 历史摘要压缩（YTY方向19深化）
        if history and isinstance(history, list):
            history, timing['history_compress'] = self._timed_call(
                self.history_manager.compress, history
            )
        else:
            timing['history_compress'] = 0

        # 1.5 上下文感知查询解析（YTY方向9）
        resolved_query = user_query
        timing['context_resolution'] = 0
        if need_retrieval and history:
            history_list = history if isinstance(history, list) else ([history] if isinstance(history, dict) else [])
            if history_list:
                resolved_query, timing['context_resolution'] = self._timed_call(
                    self.query_context_resolver.resolve, user_query, history_list
                )
                if resolved_query != user_query:
                    print(f"[QueryContextResolver] 查询已解析\n  原始: {user_query}\n  解析后: {resolved_query}")

        intent_detection_result = None
        intent_expansion_result = None
        timing['intent_detection'] = 0
        timing['intent_expansion'] = 0

        if need_retrieval:
            if enable_expansion:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_detect = executor.submit(
                        self._timed_call, self.intent_detector.detect, resolved_query
                    )
                    future_expand = executor.submit(
                        self._timed_call, self.intent_expander.expand, resolved_query
                    )
                    intent_detection_result, timing['intent_detection'] = future_detect.result()
                    intent_expansion_result, timing['intent_expansion'] = future_expand.result()
            else:
                intent_detection_result, timing['intent_detection'] = self._timed_call(
                    self.intent_detector.detect, resolved_query
                )

        timing['query_processing_total'] = time.time() - query_processing_start

        # 直接回答
        if not need_retrieval:
            for chunk in self.generator.generate_stream(
                user_query, need_retrieval=False, today=today, history=history
            ):
                yield {"type": "chunk", "content": chunk}

            timing['total'] = time.time() - total_start
            yield {
                "type": "done",
                "data": {
                    "references": {"comments": [], "summaries": []},
                    "timing": timing
                }
            }
            return

        # 二、混合检索
        if enable_ranking:
            final_topk_for_retrieval = retrieval_topk
        else:
            final_topk_for_retrieval = ranking_topk

        rewritten_queries = (intent_expansion_result
                             if intent_expansion_result
                             else [{'query': resolved_query, 'weight': 1.0}])

        comments, summaries, retrieval_timing, hyde_results = self.retriever.retrieve(
            rewritten_queries,
            room_type=intent_detection_result.get('room_type'),
            fuzzy_room_type=intent_detection_result.get('fuzzy_room_type'),
            topk=route_topk,
            final_topk=final_topk_for_retrieval,
            enable_bm25=enable_bm25,
            enable_vector=enable_vector,
            enable_reverse=enable_reverse,
            enable_hyde=enable_hyde,
            enable_summary=enable_summary
        )
        timing['retrieval'] = retrieval_timing

        # 三、排序
        if enable_ranking:
            ranker = MultiFactorRanker(
                self.reranker,
                embedding_client=self.embedding_client,
                w_relevance=w_relevance, w_quality=w_quality,
                w_length=w_length, w_review=w_review,
                w_useful=w_useful, w_recency=w_recency,
                base_decay=base_decay, implied_boost=implied_boost,
                clear_boost=clear_boost, half_life_days=half_life_days
            )
            ranked_comments, ranking_timing = ranker.rank(
                user_query, comments,
                time_sensitivity=intent_detection_result.get('time_sensitivity'),
                topk=ranking_topk, today=today,
                enable_diversity=enable_diversity,
                diversity_strategy=diversity_strategy,
                diversity_lambda=diversity_lambda,
                diversity_pool_size=diversity_pool_size
            )
            timing['ranking'] = ranking_timing
        else:
            ranked_comments = comments
            timing['ranking'] = {'total': 0, 'rerank': 0, 'scoring': 0, 'diversity': 0}

        # 发送参考评论
        processed_comments = []
        for c in ranked_comments:
            comment_data = {
                'comment_id': c['comment_id'],
                'comment': c['comment'],
                'metadata': c['metadata']
            }
            if enable_ranking:
                comment_data['final_rank'] = c['final_rank']
                comment_data['final_score'] = c['final_score']
                comment_data['pre_diversity_rank'] = c.get('pre_diversity_rank')
                comment_data['diversity_strategy'] = c.get('diversity_strategy')
            processed_comments.append(comment_data)

        yield {
            "type": "references",
            "data": {
                "comments": processed_comments,
                "summaries": [
                    {"summary": s['summary'], "metadata": s['metadata']}
                    for s in summaries
                ]
            }
        }

        # 四、流式回复生成
        for chunk in self.generator.generate_stream(
            user_query,
            rewritten_queries=intent_expansion_result,
            ranked_comments=ranked_comments,
            summaries=summaries,
            need_retrieval=True,
            today=today,
            history=history
        ):
            yield {"type": "chunk", "content": chunk}

        timing['total'] = time.time() - total_start
        yield {
            "type": "done",
            "data": {"timing": timing}
        }

    def _timed_call(self, func, *args) -> tuple:
        """带计时的函数调用"""
        start = time.time()
        result = func(*args)
        return result, time.time() - start
