"""回复生成器：基于检索上下文生成最终回复"""

import json
import time
from datetime import datetime
from dashscope import Generation
from modules.clients import LLMClient


class ResponseGenerator:
    """回复生成器：基于检索上下文生成最终回复"""

    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.api_key = api_key
        self.model = model
        # 方向16深化：用于回复质量自评估的轻量客户端（qwen-flash，省成本）
        self._eval_client = LLMClient(api_key, model="qwen-flash", json=True)

    def _build_prompt(self, user_query: str, rewritten_queries=None,
                      ranked_comments=None, summaries=None,
                      need_retrieval: bool = True, today: datetime | None = None,
                      history: list | dict | None = None) -> str:
        """构建生成 prompt"""

        # 构建对话历史上下文（方向19：支持多轮 list[dict] 和单轮 dict 向后兼容）
        history_context = ""
        if isinstance(history, list) and history:
            history_context = "【对话历史】\n"
            turn_num = 1
            for entry in history:
                if isinstance(entry, dict):
                    if entry.get("role") == "summary":
                        # 摘要条目：较早历史的压缩结果（方向19深化）
                        history_context += f"【历史摘要】{entry.get('content', '')}\n\n"
                    elif entry.get("user") and entry.get("assistant"):
                        # 普通轮次
                        asst = entry["assistant"]
                        if len(asst) > 300:
                            asst = asst[:300] + "..."
                        history_context += f"第{turn_num}轮\n用户：{entry['user']}\n助手：{asst}\n\n"
                        turn_num += 1
        elif isinstance(history, dict) and history.get("user") and history.get("assistant"):
            # 向后兼容：单轮历史
            history_context = f"""\n【上一轮对话】\n用户：{history['user']}\n助手：{history['assistant']}\n"""

        if not need_retrieval:
            return f"""
你是广州花园酒店的智能客服助手。

{history_context}

用户问题：{user_query}

请直接回答用户的问题。注意：
- 如果是问候或闲聊，友好回应
- 如果是通用问题，给出简洁准确的回答
- 如果用户的问题是对上一轮对话的追问，请结合上下文理解用户意图
- 语气要亲切专业
- 使用Markdown格式输出，不得出现 "```markdown", "```" 标记
"""

        if not today:
            today = datetime.today()
        date = f"{today.year}年{today.month}月{today.day}日"

        # 构建改写 Query 上下文
        queries_context = ""
        if rewritten_queries:
            queries_context += "【问题解析】\n系统识别到用户可能关注以下方面：\n"
            queries_context += "\n".join(
                [f"- {q['query']}（意图权重为{q['weight']}）" for q in rewritten_queries]
            )
            queries_context += "\n注意：权重信息是用来帮助你区分意图主次的，**不得**向用户输出权重相关信息。"

        # 构建评论上下文
        if ranked_comments:
            comments_context = "【相关用户评论】\n"
            for i, c in enumerate(ranked_comments, 1):
                comments_context += f"""
【评论{i}】
评分: {c['metadata']['score']}（满分5分）
发布日期: {c['metadata']['publish_date']}
评论文本: {c['comment']}
点赞数: {c['metadata']['useful_count']}
评论数: {c['metadata']['review_count']}
房型: {c['metadata']['room_type']}
"""
                # 优化2：附加匹配的分块，帮助 LLM 精确定位引用
                chunks = c.get('matched_chunks', [])
                if chunks:
                    comments_context += "【匹配原文片段】\n"
                    for ch in chunks[:2]:
                        comments_context += f"  · \"{ch['text'][:150]}\"\n"
        else:
            comments_context = "【未检索到相关用户评论】\n"

        # 构建摘要上下文
        summaries_context = ""
        if summaries:
            summaries_context += "【相关评论摘要】\n"
            for s in summaries:
                summaries_context += f"""
【{s['metadata']['category']}类别摘要】
关键词: {s['metadata']['keywords']}
摘要: {s['summary']}
"""
            summaries_context += """
注意：评论摘要是用来给到你更丰富的概览信息的，但用户只能看到【相关用户评论】的引用而看不到摘要的引用，因此在回复中你可以给出摘要中的模糊信息，\
但**不得过于精确因为用户无法溯源**，也**不得告诉用户你引用了摘要**，**更不得将其当作评论引用输出"评论x"**。若摘要中的信息与用户问题无关，直接忽略即可，**不需要**做出任何额外说明。
"""

        return f"""
你是广州花园酒店的智能客服助手，需要基于用户评论为用户提供准确、高质量、有帮助、简洁的回答。

今天是：{date}

{history_context}

用户问题：{user_query}

{queries_context}

{comments_context}

{summaries_context}

【回答要求】
1. 综合以上评论信息，给出客观、全面的回答
2. 回答要有条理，突出重点：
   - 若问题涉及**多个独立方面**（如“早餐和停车场怎么样”、“房间和服务分别如何”），使用 Markdown 二级标题（`##`）分段组织回答，可先用 1−2 句话给出整体概括，再按方面展开细节
   - 若问题聚焦**单一方面**，给出流畅的段落式回答，无需强行分段
3. 如有正面和负面评价，都要提及，保持客观。注意给出的参考评论并不代表所有，切忌以偏概全给出“绝对化”的表述
4. 语气要专业、亲切
5. 回答长度适中，不要过于冗长
6. 不得大段或连续照抄用户评论，严禁全文都在引用用户评论却并没有思考提炼总结。相似内容能合并就合并，不要分开引用（合并后注意不得同时列出超过3条参考评论，使用"等"替代）
7. 一般来说越靠前的评论，其重要性越高，但你也可以自行判断自行选择
8. 不得在回复中罗列用户评论的具体日期，但当用户问题时效性敏感时，可以大致提一下参考评论的时间范围；当用户未表现出明显时效性需求时不要强行给出具体时间
9. 引用【相关用户评论】中某一条评论独特内容时，应使用引用标记 [[ref:N]]（N为评论序号）标注来源（**仅标注非常确定的引用，模棱两可的引用不要标注，务必保证引用序号绝对正确**），供用户参考；但针对参考评论总体（如"多数住客……"等内容）或【xx类别摘要】进行归纳总结时**无需**标注。引用标记示例：某某服务很好[[ref:2]]。不要在标记外面加任何括号或其他包裹符号
10. 不得同时列出超过3条引用，即最多 [[ref:1,3,5]]。如需同时引用超过3条评论，则应只保留排名最靠前的2条并加"等"字，输出形式为 [[ref:1,3]]等。注意多条引用写在同一个标记内用逗号分隔，如 [[ref:1,3]]，而不是 [[ref:1]][[ref:3]]
11. 如果评论信息不足以回答问题，诚实说明
12. 所有的回复必须仅依赖检索到的用户评论及摘要，不得出现自作主张的幻觉回复，例如帮用户查询酒店今日客房剩余、当前酒店相关活动推荐等一律不允许出现。你并没有接入酒店内部API无法完成这些事情因此禁止在回复中出现此类幻觉信息
13. 使用Markdown格式输出，不得出现 "```markdown", "```" 标记

用户问题：{user_query}

请给出你的回答：
"""

    def evaluate_response(self, user_query: str, response: str,
                           ranked_comments: list | None = None) -> dict:
        """
        回复质量自评估（方向16深化）

        返回:
            {
                "hallucination_risk": 0-1,   # 幻觉风险
                "structure_score": 1-5,      # 结构清晰度
                "citation_accuracy": 0-1,    # 引用准确度
                "overall": 1-5,              # 综合评分
                "reason": "..."
            }
        """
        comments_summary = ""
        if ranked_comments:
            for i, c in enumerate(ranked_comments[:5], 1):
                snippet = c.get('comment', '')[:100]
                comments_summary += f"评论{i}: {snippet}\n"

        prompt = f"""你是一个RAG系统回复质量评估专家。请对下面的智能客服回复进行质量评分。

【用户问题】
{user_query}

【参考评论（部分）】
{comments_summary if comments_summary else '（无参考评论）'}

【待评估的回复】
{response[:800]}

请以JSON格式输出以下评分：
{{
    "hallucination_risk": 0到1之间浮点数，0无幻觉1严重幻觉（出现参考评论中未提及的具体信息）,
    "structure_score": 1到5整数，5结构清晰有条理，1杂乱无章,
    "citation_accuracy": 0到1之间浮点数，1引用准确对应评论内容，0引用严重错误,
    "overall": 1到5整数，综合质量评分,
    "reason": "不超过30字的简短说明"
}}"""

        default_result = {"hallucination_risk": 0.0, "structure_score": 3,
                          "citation_accuracy": 1.0, "overall": 3, "reason": "评估失败"}

        for i in range(2):
            try:
                raw = self._eval_client.generate(prompt, temperature=0.1)
                raw = raw.replace('```json', '').replace('```', '').strip()
                data = json.loads(raw)
                return {
                    "hallucination_risk": max(0.0, min(1.0, float(data.get("hallucination_risk", 0.0)))),
                    "structure_score": max(1, min(5, int(data.get("structure_score", 3)))),
                    "citation_accuracy": max(0.0, min(1.0, float(data.get("citation_accuracy", 1.0)))),
                    "overall": max(1, min(5, int(data.get("overall", 3)))),
                    "reason": str(data.get("reason", ""))
                }
            except Exception as e:
                print(f"回复质量评估第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("回复质量评估失败，返回默认分")
        return default_result

    def _call_kwargs(self, prompt: str, temperature: float = 0.7) -> dict:
        """构建 Generation.call() 的通用参数"""
        return dict(
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            result_format="message",
            stream=True,
            incremental_output=True
        )

    def generate(self, user_query: str, rewritten_queries=None, ranked_comments=None,
                 summaries=None, need_retrieval: bool = True, print_response: bool = True,
                 today: datetime | None = None, history: list | dict | None = None,
                 enable_evaluation: bool = True) -> tuple[str, float, float, float, dict]:
        """
        生成回复（流式输出）

        返回:
            (response_text, ttft_model, subsequent_time, generation_time, evaluation)
            evaluation 为质量评估结果，当 hallucination_risk > 0.7 或 overall < 3 时自动重试一次
        """
        start_time = time.time()
        prompt = self._build_prompt(user_query, rewritten_queries, ranked_comments,
                                    summaries, need_retrieval, today, history)

        completion = Generation.call(**self._call_kwargs(prompt))

        response_content = ""
        ttft_model = 0
        subsequent_time = 0
        first_token_time = 0

        for chunk in completion:
            if chunk.status_code != 200:
                raise RuntimeError(f"回复生成失败: {chunk.message}")

            message = chunk.output.choices[0].message
            if message.content:
                if not ttft_model:
                    ttft_model = time.time() - start_time
                    first_token_time = time.time()
                if print_response:
                    print(message.content, end="", flush=True)
                response_content += message.content

        if print_response and response_content:
            print()

        if ttft_model:
            subsequent_time = time.time() - first_token_time

        generation_time = time.time() - start_time

        # 方向16深化：回复质量自评估
        evaluation = {}
        if enable_evaluation and need_retrieval and response_content:
            evaluation = self.evaluate_response(user_query, response_content, ranked_comments)
            # 当幻觉风险过高或评分过低时，自动重新生成一次
            if evaluation.get('hallucination_risk', 0) > 0.7 or evaluation.get('overall', 5) < 3:
                print(f"[自评估] 回复质量不赠（overall={evaluation.get('overall')}, "
                      f"hallucination={evaluation.get('hallucination_risk'):.2f}），正在重新生成...")
                retry_completion = Generation.call(**self._call_kwargs(prompt))
                retry_content = ""
                for chunk in retry_completion:
                    if chunk.status_code == 200:
                        message = chunk.output.choices[0].message
                        if message.content:
                            retry_content += message.content
                if retry_content:
                    response_content = retry_content
                    # 对重新生成的回复再次评估
                    evaluation = self.evaluate_response(user_query, response_content, ranked_comments)
                    evaluation['retried'] = True
                    print(f"[自评估] 重新生成完成（overall={evaluation.get('overall')}")

        return response_content, ttft_model, subsequent_time, generation_time, evaluation

    def generate_stream(self, user_query: str, rewritten_queries=None, ranked_comments=None,
                        summaries=None, need_retrieval: bool = True,
                        today: datetime | None = None, history: list | dict | None = None):
        """
        流式生成回复（yield 每个 chunk）

        Yields:
            str: 每个文本 chunk
        """
        prompt = self._build_prompt(user_query, rewritten_queries, ranked_comments,
                                    summaries, need_retrieval, today, history)

        completion = Generation.call(**self._call_kwargs(prompt))

        for chunk in completion:
            if chunk.status_code != 200:
                raise RuntimeError(f"回复生成失败: {chunk.message}")

            message = chunk.output.choices[0].message
            if message.content:
                yield message.content
