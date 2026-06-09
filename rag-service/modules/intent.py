"""意图处理模块：识别、检测、扩展、HyDE 生成"""

import json
import time
from dashscope import Generation


class IntentRecognizer:
    """意图识别器：判断问题是否需要检索知识库"""

    def __init__(self, api_key: str, model: str = "qwen-flash"):
        self.api_key = api_key
        self.model = model

    def recognize(self, query: str, **kwargs) -> str:
        """识别用户意图，返回 True 表示需要检索"""
        system_prompt = """你是广州花园酒店的意图分类器。根据用户的问题，判断是否需要检索酒店评论知识库。

分类规则：
- RETRIEVAL：问题涉及酒店的设施、服务、房间、位置、餐饮、价格、体验等具体信息，需要检索评论才能回答
- DIRECT：问候、闲聊、常识性问题等，不涉及该酒店的具体信息，可以直接回答

只回复 RETRIEVAL 或 DIRECT，不要输出任何其他内容。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            result_format="message"
        )

        if response.status_code == 200:
            intent = response.output.choices[0].message.content.strip()
            return intent == "RETRIEVAL"
        else:
            raise RuntimeError(f"意图识别失败: {response.message}")


class IntentDetector:
    """意图检测器：提取房型约束与时效性需求"""

    def __init__(self, llm_client, exact_room_types: list, fuzzy_room_types: list):
        self.llm_client = llm_client
        self.exact_room_types = exact_room_types
        self.fuzzy_room_types = fuzzy_room_types

    def detect(self, query: str) -> dict:
        """
        检测用户意图

        返回:
            {
                "room_type": "花园大床房" | ... | None,
                "fuzzy_room_type": "大床房" | ... | None,
                "time_sensitivity": "clear" | "implied" | None
            }
        """
        prompt = f"""
你是一个酒店智能客服助手，需要分析用户查询并提取关键信息。

【任务】
从用户查询中提取以下信息：
1. 房型约束：用户是否提到特定房型
2. 时效性需求：用户是否关注最新信息

【精确房型列表】
{json.dumps(self.exact_room_types, ensure_ascii=False)}

【模糊房型列表】
{json.dumps(self.fuzzy_room_types, ensure_ascii=False)}

【房型检测规则】
- 优先检测精确房型，如检测到则填入 room_type，若模棱两可或只能检测到模糊房型则视为未检测到，填入 None。填入的内容只能是【精确房型列表】中的房型名称或 None
- 如未检测到精确房型，尝试检测模糊房型，如检测到则填入 fuzzy_room_type，若模棱两可则视为未检测到，填入 None。填入的内容只能是【模糊房型列表】中的房型名称或 None
- 如都未检测到，两者均为 None

【时效性判断标准】
- clear: 用户明确提到"最近"、"今年"、"最新"、"现在"等词汇
- implied: 用户隐含关注当前现状，但未明确表达，表现弱时效性
- None: 用户未表现出时效性关注

【用户查询】
{query}

【输出格式】
严格以 JSON 格式输出：
{{
    "room_type": "花园大床房" 或 None,
    "fuzzy_room_type": "大床房" 或 None,
    "time_sensitivity": "clear" 或 "implied" 或 None
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                if data['room_type'] and data['room_type'] not in self.exact_room_types:
                    data['room_type'] = None
                if data['fuzzy_room_type'] and data['fuzzy_room_type'] not in self.fuzzy_room_types:
                    data['fuzzy_room_type'] = None
                if data['time_sensitivity'] and data['time_sensitivity'] not in ['clear', 'implied']:
                    data['time_sensitivity'] = None
                return data
            except Exception as e:
                print(f"意图检测第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("意图检测失败，已返回全 None 字典")
        return {
            "room_type": None,
            "fuzzy_room_type": None,
            "time_sensitivity": None
        }


class IntentExpander:
    """意图扩展器：改写 Query 并计算权重；自动识别比较型查询并走专用分解路径（方向9深化）"""

    # 触发比较型判断的关键词（必须如实匹配，不做语义拓展）
    _COMPARATIVE_KEYWORDS = ['哪个', '哪些', '比较', '对比', 'vs', 'VS', '更好', '更山', '更担心',
                              '分别', '各自', '哪种', '与', '和…哪个', '还是', '对比一下']

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def is_comparative_query(self, query: str) -> bool:
        """
        判断查询是否为比较型/评估型（方向9深化）

        策略：先用关键词规则判断（快速且免费），
        分析含多个明确实体的查询也判定为比较型。
        """
        for kw in self._COMPARATIVE_KEYWORDS:
            if kw in query:
                return True
        # 如果包含〄00」和〄00」以外的实体，如 A和B哪个更好
        # 简单启发式：含有连词“和”且包含问号的可诚判为比较型
        if '和' in query and '?' in query or '和' in query and '？' in query:
            return True
        return False

    def decompose_for_comparison(self, query: str) -> list:
        """
        针对比较型查询进行专用分解（方向9深化）

        将比较对象各自拆为独立子查询，权重均分。
        返回格式与 expand() 相同。
        """
        prompt = f"""你是一个查询分解助手，专门处理包含比较/多方面的查询。

【用户查询】
{query}

【任务】
将上述查询分解为 2-4 个独立子查询，每个子查询聊焦一个具体方面。
权重均分，之和为 1，且只能是 0.2 的倍数。

【输出格式】
严格以 JSON 格式输出：
{{
    "sub_queries": [
        {{"query": "子查询1", "weight": 0.5}},
        {{"query": "子查询2", "weight": 0.5}}
    ]
}}"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.2)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                queries = data.get('sub_queries', [])
                if isinstance(queries, list) and len(queries) >= 2:
                    return [{'query': q['query'], 'weight': float(q['weight'])} for q in queries]
            except Exception as e:
                print(f"比较型查询分解第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("比较型查询分解失败，回退到原始查询")
        return [{'query': query, 'weight': 1.0}]

    def _expand_normal(self, query: str) -> list | None:
        """原有扩展逻辑（重命名为内部方法）"""
        prompt = f"""
你是一个酒店智能客服助手，需要深度理解用户查询意图。

【任务】
1. 分析用户查询，检测用户的核心关注点
2. 生成1-3个改写后的查询，每个查询更清晰、更具体地表达一个关注点
3. 为每个改写查询分配权重，表示该关注点的重要性（权重之和为1，且只允许使用0.2的倍数，儸0.2,0.4,0.6,0.8,1.0）

【用户查询】
{query}

【要求】
- 改写的查询应该比原查询更具体、更明确
- 每个改写查询应该聚焦一个具体方面
- 权重应该反映该方面在原查询中的重要性
- 对于模糊的查询，使用尽可能多的改写来覆盖更大范围的意图；对于明确的查询，不要对其过度展开

【输出格式】
严格以 JSON 格式输出：
{{
    "rewritten_queries": [
        {{"query": "酒店交通是否便利？", "weight": 0.6}},
        {{"query": "酒店周边有哪些配套设施？", "weight": 0.2}},
        {{"query": "酒店的服务效率如何？", "weight": 0.2}}
    ]
}}

【注意】
- rewritten_queries 数组长度为1-3
- 所有 weight 之和必须等于1，且只允许使用0.2的倍数
"""
        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.3)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                queries = data['rewritten_queries']
                if isinstance(queries, list):
                    for item in queries:
                        item['query'] = item['query']
                        item['weight'] = float(item['weight'])
                    return queries
                else:
                    raise TypeError(f"queries 数据类型错误: 期望 list, 实际为 {type(queries).__name__}")
            except Exception as e:
                print(f"意图扩展第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue
        print("意图扩展失败，已返回 None")
        return None

    def expand(self, query: str) -> list | None:
        """
        扩展用户意图（已升级：自动识别比较型查询并走专用分解路径）

        返回:
            [
                {"query": "改写的查询1", "weight": 0.6},
                {"query": "改写的查询2", "weight": 0.2},
                {"query": "改写的查询3", "weight": 0.2}
            ]
        """
        # 方向9深化：比较型查询走专用分解路径
        if self.is_comparative_query(query):
            print(f"[比较型查询] 检测到比较意图，走专用分解路径: {query}")
            return self.decompose_for_comparison(query)
        # 普通查询走原有扩展路径
        return self._expand_normal(query)


class HyDEGenerator:
    """假设性回复生成器：为单个 Query 生成假设回复用于增强检索"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, query: str) -> list[str]:
        """
        为单个查询生成假设性回复

        策略：生成2条正面回复 + 1条负面回复
        """
        prompt = f"""
你是一个酒店评论撰写者，需要为以下查询生成假设性的评论回复。

【查询】
{query}

【任务】
针对上述查询，生成3条假设性的酒店评论：
- 2条正面评论：积极评价酒店相关方面
- 1条负面评论：指出可能存在的不足

【要求】
- 每条评论50-100字
- 评论要具体、真实，包含细节
- 评论风格要像真实用户写的
- 尽量增大3条评论之间的差异性

【输出格式】
严格以 JSON 格式输出：
{{
    "hypothetical_responses": [
        "正面评论1",
        "正面评论2",
        "负面评论"
    ]
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.7)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                responses = data['hypothetical_responses']
                if isinstance(responses, list):
                    return responses
                else:
                    raise TypeError(f"responses 数据类型错误: 期望 list, 实际为 {type(responses).__name__}")
            except Exception as e:
                print(f"假设性回复生成第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("假设性回复生成失败，已返回原查询")
        return [query]


class QueryContextResolver:
    """上下文感知查询解析器：将含指代/省略的追问改写为完整独立查询（方向9：复杂Query理解）"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def resolve(self, query: str, history: list) -> str:
        """
        根据对话历史，将可能含有上下文引用的查询改写为完整的独立查询。
        若查询已是完整独立的，原样返回。

        参数:
            query: 用户当前输入的查询
            history: 对话历史，格式为 [{"user": "...", "assistant": "..."}, ...]
        返回:
            str: 改写后的独立查询（或原始查询）
        """
        if not history:
            return query

        # 构建最近3轮历史文本（截断长回复，控制 token 消耗）
        history_text = ""
        for i, turn in enumerate(history[-3:], 1):
            if isinstance(turn, dict) and turn.get("user") and turn.get("assistant"):
                assistant_text = turn["assistant"]
                if len(assistant_text) > 200:
                    assistant_text = assistant_text[:200] + "..."
                history_text += f"第{i}轮 - 用户：{turn['user']}\n第{i}轮 - 助手：{assistant_text}\n"

        if not history_text:
            return query

        prompt = f"""你是一个查询解析助手。根据对话历史，将当前查询改写为完整独立的查询（无需上下文也能理解）。

【对话历史】
{history_text}
【当前查询】
{query}

【改写规则】
1. 如果当前查询含有指代词（如"它"、"那"、"这"、"同样的"等）、省略成分、追问语气（如"那xx呢?"、"还有呢?"、"怎么样?"），需要结合历史补充完整
2. 如果当前查询完整清晰、与历史对话无明显关联，直接返回原查询
3. 改写时保持用户的原始意图，只补充必要的上下文信息，不要过度扩展
4. 改写后的查询应简洁，不超过50字

【输出格式】
只输出改写后的查询文本，不要任何解释、标点前缀或额外内容。"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)
                resolved = response.strip().replace('```', '').strip()
                if resolved and len(resolved) > 3:
                    return resolved
                return query
            except Exception as e:
                print(f"查询上下文解析第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("查询上下文解析失败，返回原始查询")
        return query
