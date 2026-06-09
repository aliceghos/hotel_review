# 酒店评论智能问答系统

**Hotel Review Intelligent Q\&A System** —— 基于 RAG 的混合检索与多维度优化

《大模型技术原理与商业应用》课程期末小组项目。以**广州花园酒店**真实住客评论（携程，2,171 条）为数据基础，支持评论浏览与 AI 自然语言问答。

## 小组成员与分工

| 成员 | 负责内容 | 优化方向 |
|------|----------|----------|
| **韩锐研** | 优化 5、6、12 | Doc2Query 小模型替代 / 多粒度摘要生成与检索 / 多模态检索 |
| **陈星宇** | 项目整合 + 优化 2、4 | 评论文本分块与引用高亮 / 聚类类别划分与多标签 |
| **姚天瑜** | 优化 9、16、19 | 复杂 Query 理解与意图改写 / 结构化回复 Prompt 优化 / 多轮对话上下文管理 |
| **程思伟** | 优化 7、8、15 | Elasticsearch 倒排索引 / 向量检索引擎抽象层 / 多样性重排 MMR/DPP |

## 技术架构

```
Browser (Next.js 15)
  ├── 评论浏览 ← Insforge PostgreSQL
  └── AI 问答 (SSE) → FastAPI (:8000) → RAG Pipeline
                           │
                    ┌──────┴──────────┐
                    │  DashVector     │  (云端向量库)
                    │  ChromaDB       │  (本地向量库 + 分块/摘要)
                    │  BM25 (jieba)   │  (关键词检索)
                    │  Qwen3-Rerank   │  (重排序)
                    │  Qwen-Plus      │  (回复生成)
                    └─────────────────┘
```

| 层级 | 技术栈 |
|------|--------|
| 前端 | Next.js 15 · React · TypeScript · Tailwind CSS |
| 后端 BaaS | Insforge (PostgreSQL) |
| LLM 框架 | DashScope (Qwen-Plus / Qwen-Flash) |
| 向量库 | DashVector (云端) · ChromaDB (本地) |
| 检索 | text-embedding-v4 · BM25 (jieba) · Qwen3-Rerank |
| 嵌入 | text-embedding-v4 (1024 维) |

## 功能概览

### 评论浏览
- 卡片展示评论（星级、房型、出行类型、图片）
- 多条件筛选：评分 / 房型 / 出行类型 / 类别 / 关键词搜索
- 排序：发布日期 / 评分 / 质量分 / 有用数
- 无限滚动分页、图片灯箱
- 统计面板：总评论数、平均评分、含图评论数

### AI 智能问答
- 浮动聊天面板，自然语言提问
- 六路混合检索：BM25 + 向量 + 反向 Query + HyDE + 摘要 + 分块
- RRF 融合 → Qwen3-Rerank → 多因子评分 → MMR/DPP 多样性重排
- 引用标记 `[N]` 悬停可查看原文片段
- 多轮对话记忆，流式输出 (SSE)

## 项目结构

```
merged/
├── rag-service/                 # Python 后端
│   ├── main.py                  # FastAPI 入口 (port 8000)
│   ├── config.py                # 常量配置
│   ├── requirements.txt         # Python 依赖
│   ├── test_rag.py              # 验证脚本
│   ├── modules/                 # 核心模块
│   │   ├── clients.py           # LLM/Embedding 客户端
│   │   ├── index.py             # 本地 BM25 倒排索引
│   │   ├── text_search.py       # ES BM25 (可选)
│   │   ├── vector_store.py      # 向量库抽象层
│   │   ├── intent.py            # 意图识别/检测/扩展/上下文解析
│   │   ├── retriever.py         # 六路混合检索 + RRF 融合
│   │   ├── ranker.py            # Rerank + 多因子 + MMR/DPP
│   │   ├── generator.py         # 回复生成 + 自评估
│   │   ├── history_manager.py   # 多轮对话历史压缩
│   │   ├── chunking.py          # 文本分块 + 分块索引
│   │   ├── clustering.py        # KMeans 聚类 + 软多标签
│   │   └── rag_system.py        # RAG 主控工作流
│   ├── hry/                     # 韩锐研优化模块 (5/6/12)
│   ├── scripts/                 # 离线构建脚本
│   ├── data/                    # 数据文件
│   └── utils/                   # 工具函数
├── src/                         # Next.js 前端
│   ├── app/                     # 页面 + API 路由
│   ├── components/              # UI 组件
│   └── lib/                     # 类型/API 调用
├── public/                      # 静态资源
└── package.json                 # Node 依赖
```

## 快速启动

### 0. 环境要求
- Python 3.10+ · Node.js 18+ · DashScope API Key · DashVector 实例

### 1. 安装后端依赖
```bash
cd rag-service
pip install -r requirements.txt
```

### 2. 配置环境变量
将 `rag-service/.env.example` 复制为 `rag-service/.env`，填入你的 API Key：

```ini
DASHSCOPE_API_KEY=sk-xxx
DASHVECTOR_API_KEY=sk-xxx
DASHVECTOR_HOTEL_ENDPOINT=vrs-cn-xxx.dashvector.cn-hangzhou.aliyuncs.com
```

### 3. (可选) 构建离线索引
```bash
# 构建评论分块索引
python scripts/build_chunks.py

# 构建聚类类别系统
python scripts/build_clusters.py
```

### 4. 启动后端服务
```bash
cd rag-service
python main.py
# → http://localhost:8000
```

### 5. 启动前端
```bash
npm install
npm run dev
# → http://localhost:3000
```

### 6. 验证测试
```bash
cd rag-service
# 导入验证
python -X utf8 test_rag.py
# 端到端测试 (需要有效 API Key)
python -X utf8 test_rag.py --full
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/chat` | POST | RAG 流式问答 (SSE) |

请求格式：
```json
{
  "query": "酒店的早餐怎么样？",
  "options": { "enable_hyde": false },
  "history": [{"user": "...", "assistant": "..."}]
}
```

## 优化功能清单

| 编号 | 优化方向 | 负责 | 关键文件 |
|------|----------|------|----------|
| 2 | 评论文本分块 + 引用高亮 | 陈星宇 | `modules/chunking.py`, `ChatWidget.tsx` |
| 4 | 聚类类别划分 + 多标签 | 陈星宇 | `modules/clustering.py`, `scripts/build_clusters.py` |
| 5 | Doc2Query 小模型替代 | 韩锐研 | `hry/doc2query_model.py` |
| 6 | 多粒度摘要生成与检索 | 韩锐研 | `hry/multi_granularity_summary.py` |
| 12 | 多模态检索 | 韩锐研 | `hry/multimodal_retriever.py` |
| 7 | Elasticsearch 倒排索引 | 程思伟 | `modules/text_search.py` |
| 8 | 向量检索引擎抽象层 | 程思伟 | `modules/vector_store.py` |
| 15 | MMR/DPP 多样性重排 | 程思伟 | `modules/ranker.py` (DiversityReranker) |
| 9 | 复杂 Query 理解与意图改写 | 姚天瑜 | `modules/intent.py` (QueryContextResolver) |
| 16 | 结构化回复 Prompt + 自评估 | 姚天瑜 | `modules/generator.py` |
| 19 | 多轮对话上下文管理 | 姚天瑜 | `modules/history_manager.py`, `qa-background.ts` |

## 许可证

课程作业，仅供学习参考。
