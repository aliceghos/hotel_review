"""RAG 系统验证脚本 — 验证所有模块导入正常

运行方式: cd rag-service && python -X utf8 test_rag.py
"""

import os
import sys
from pathlib import Path

# 确保从 rag-service 目录运行
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """验证所有模块可以正常导入"""
    print("=" * 60)
    print("1. 测试模块导入")
    print("=" * 60)

    from config import TODAY, EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES
    print(f"  ✅ config: TODAY={TODAY}, {len(EXACT_ROOM_TYPES)} 精确房型, {len(FUZZY_ROOM_TYPES)} 模糊房型")

    from modules.clients import LLMClient, EmbeddingClient
    print(f"  ✅ clients: LLMClient, EmbeddingClient")

    from modules.index import InvertedIndex
    print(f"  ✅ index: InvertedIndex (本地 BM25)")

    from modules.text_search import ElasticsearchBM25Index
    print(f"  ✅ text_search: ElasticsearchBM25Index (可选)")

    from modules.vector_store import DashVectorSearchEngine, create_vector_search_engine
    print(f"  ✅ vector_store: DashVectorSearchEngine, create_vector_search_engine")

    from modules.intent import (
        IntentRecognizer, IntentDetector, IntentExpander,
        HyDEGenerator, QueryContextResolver
    )
    print(f"  ✅ intent: IntentRecognizer, IntentDetector, IntentExpander, HyDEGenerator, QueryContextResolver")

    from modules.retriever import HybridRetriever
    print(f"  ✅ retriever: HybridRetriever")

    from modules.ranker import Reranker, MultiFactorRanker, DiversityReranker
    print(f"  ✅ ranker: Reranker, MultiFactorRanker, DiversityReranker (MMR/DPP)")

    from modules.generator import ResponseGenerator
    print(f"  ✅ generator: ResponseGenerator (结构化回复 + 自评估)")

    from modules.history_manager import ConversationHistoryManager
    print(f"  ✅ history_manager: ConversationHistoryManager (历史压缩)")

    from modules.rag_system import HotelReviewRAG
    print(f"  ✅ rag_system: HotelReviewRAG (集成所有优化)")

    from utils.formatting import print_retrieval_results, print_rag_result
    print(f"  ✅ formatting: 格式化输出")

    from utils.database import get_all_comments_from_insforge
    print(f"  ✅ database: Insforge 数据加载")

    print(f"\n  所有核心模块导入成功！")

    # 测试 HRY 模块导入
    print(f"\n  测试 HRY 优化模块...")
    try:
        from hry import doc2query_model
        print(f"    ✅ doc2query_model")
    except Exception as e:
        print(f"    ⚠️ doc2query_model 导入失败: {e}")

    try:
        from hry import adaptive_retriever
        print(f"    ✅ adaptive_retriever")
    except Exception as e:
        print(f"    ⚠️ adaptive_retriever 导入失败: {e}")

    try:
        from hry import multimodal_retriever
        print(f"    ✅ multimodal_retriever")
    except Exception as e:
        print(f"    ⚠️ multimodal_retriever 导入失败: {e}")

    try:
        from hry import offline_knowledge_base_enhanced
        print(f"    ✅ offline_knowledge_base_enhanced")
    except Exception as e:
        print(f"    ⚠️ offline_knowledge_base_enhanced 导入失败: {e}")

    try:
        from hry import multi_granularity_summary
        print(f"    ✅ multi_granularity_summary")
    except Exception as e:
        print(f"    ⚠️ multi_granularity_summary 导入失败: {e}")

    return True


def test_rag_system():
    """验证 RAG 系统初始化和查询（需要 API Key）"""
    print("\n" + "=" * 60)
    print("2. 测试 RAG 系统初始化与查询")
    print("=" * 60)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    dashvector_api_key = os.getenv("DASHVECTOR_API_KEY")
    dashvector_endpoint = os.getenv("DASHVECTOR_HOTEL_ENDPOINT")

    if not all([api_key, dashvector_api_key, dashvector_endpoint]):
        print("  ⚠️ 缺少 API Key 环境变量，跳过系统测试")
        return False

    data_dir = Path(__file__).parent / "data"

    if not (data_dir / "chroma_db").exists():
        print("  ⚠️ ChromaDB 数据目录不存在，跳过系统测试")
        return False

    has_es = bool(os.getenv("ELASTICSEARCH_URL"))
    has_local = (data_dir / "inverted_index.pkl").exists()
    if not has_es and not has_local:
        print("  ⚠️ 无文本索引（ES 或本地 pickle），跳过系统测试")
        return False

    from modules.rag_system import HotelReviewRAG
    from utils.formatting import print_rag_result

    index_mode = "Elasticsearch" if has_es else "本地 pickle"
    print(f"  正在初始化 RAG 系统（文本索引: {index_mode}）...")

    try:
        rag = HotelReviewRAG(
            api_key=api_key,
            dashvector_api_key=dashvector_api_key,
            dashvector_endpoint=dashvector_endpoint,
            data_dir=data_dir
        )
        print("  ✅ RAG 系统初始化成功！")
    except Exception as e:
        print(f"  ❌ RAG 系统初始化失败: {e}")
        return False

    test_query = "酒店的早餐怎么样？"
    print(f"\n  测试查询: {test_query}")
    print("-" * 60)

    try:
        result = rag.query(
            test_query,
            enable_hyde=False,
            print_response=False,
            enable_diversity=False
        )

        print("-" * 60)
        print(f"  回复: {result['response'][:200]}...")
        print(f"  参考评论数: {len(result['references']['comments'])}")
        print(f"  意图扩展: {result['query_processing']['intent_expansion']}")
        print(f"  总延迟: {result['timing']['total']:.3f}s")

        if result.get('evaluation'):
            print(f"  回复评估: overall={result['evaluation'].get('overall')}")
    except Exception as e:
        print(f"  ❌ 查询测试失败: {e}")
        return False

    print(f"\n  ✅ RAG 查询测试完成！")
    return True


if __name__ == "__main__":
    success = test_imports()

    if success and "--full" in sys.argv:
        test_rag_system()
    elif success:
        print("\n💡 运行 `python test_rag.py --full` 进行完整 RAG 系统测试")
        print("   (需要 DASHSCOPE_API_KEY, DASHVECTOR_API_KEY, DASHVECTOR_HOTEL_ENDPOINT)")
