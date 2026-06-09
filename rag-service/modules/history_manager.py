"""对话历史管理器：历史摘要压缩，避免长对话消耗过多 token（方向19深化）"""

import time
from modules.clients import LLMClient

# 默认保留的最近轮数，超出部分压缩为摘要
KEEP_RECENT_TURNS = 3


class ConversationHistoryManager:
    """
    对话历史管理器

    当历史轮数超过 keep_recent 时，把更早的轮次用 LLM（qwen-flash）压缩为一条
    摘要条目 {"role": "summary", "content": "..."}，避免长对话 prompt 过大。

    压缩策略：
    - history 长度 <= keep_recent：直接返回原列表
    - history 长度 > keep_recent：
        旧轮次（含已有摘要）→ LLM 压缩为新摘要
        返回 [summary_entry] + history[-keep_recent:]
    """

    def __init__(self, llm_client: LLMClient, keep_recent: int = KEEP_RECENT_TURNS):
        self.llm_client = llm_client
        self.keep_recent = keep_recent

    def compress(self, history: list) -> list:
        """
        压缩历史记录

        参数:
            history: 对话历史，格式为 [{"user": "...", "assistant": "..."}, ...]
                     或含有 {"role": "summary", "content": "..."} 的压缩条目
        返回:
            list: 压缩后的历史（可能含一条摘要 + 最近 N 轮普通对话）
        """
        if not history or len(history) <= self.keep_recent:
            return history

        # 分离：待压缩的旧轮次 vs 保留的最近轮次
        to_compress = history[:-self.keep_recent]
        recent = history[-self.keep_recent:]

        # 旧轮次中可能已有摘要条目，提取出来一起重新压缩
        existing_summary = ""
        normal_turns = []
        for entry in to_compress:
            if isinstance(entry, dict) and entry.get("role") == "summary":
                existing_summary = entry.get("content", "")
            elif isinstance(entry, dict) and entry.get("user") and entry.get("assistant"):
                normal_turns.append(entry)

        # 构建待压缩文本
        compress_text = ""
        if existing_summary:
            compress_text += f"[之前的摘要]\n{existing_summary}\n\n"
        for turn in normal_turns:
            asst = turn["assistant"]
            if len(asst) > 150:
                asst = asst[:150] + "..."
            compress_text += f"用户：{turn['user']}\n助手：{asst}\n"

        if not compress_text.strip():
            # 旧轮次全部无效，直接返回最近轮次
            return recent

        summary_content = self._summarize(compress_text)
        summary_entry = {"role": "summary", "content": summary_content}
        return [summary_entry] + recent

    def _summarize(self, history_text: str) -> str:
        """调用 LLM 将历史文本压缩为简洁摘要"""
        prompt = f"""你是一个对话摘要助手。请将以下酒店智能客服的对话历史压缩为简洁摘要。

【待压缩的对话历史】
{history_text}

【摘要要求】
1. 概括用户主要询问了哪些方面（如早餐、停车、房间、服务等）
2. 提及用户已获得的关键信息（如"早餐品种丰富"、"停车费较贵"）
3. 如有未解决的关注点，也要保留
4. 摘要不超过100字，语言简洁

【输出格式】
直接输出摘要文本，不要任何前缀或解释。

示例：
用户此前主要询问了酒店早餐和停车场情况，已了解早餐品种较丰富但停车费较贵，还关注了房间隔音问题。"""

        for i in range(2):
            try:
                result = self.llm_client.generate(prompt, temperature=0.3)
                result = result.strip()
                if result:
                    return result
            except Exception as e:
                print(f"历史摘要生成第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("历史摘要生成失败，使用简单 fallback")
        return "用户此前进行了多轮对话咨询。"
