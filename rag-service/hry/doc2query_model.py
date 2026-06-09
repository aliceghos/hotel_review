"""
doc2query 模型推理模块

基于规则和GPT-2两种模式为文档生成潜在查询问题。
"""

import logging
import re
import json
from typing import Optional
from typing import List

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RuleBasedGenerator:
    """基于规则的简单问题生成器"""
    
    # 问题模板
    QUESTION_TEMPLATES = [
        ("什么", "什么{}?"),
        ("如何", "如何{}?"),
        ("为什么", "为什么{}?"),
        ("哪里", "{}在哪里?"),
        ("什么时候", "{}是什么时候?"),
        ("谁", "谁是{}?"),
        ("多少", "{}有多少?"),
    ]
    
    # 关键词模式
    KEYWORD_PATTERNS = {
        r".*是.*": "这是什么?",
        r".*做.*": "怎么做?",
        r".*使用.*": "如何使用?",
        r".*原因.*": "原因是什么?",
        r".*方法.*": "有什么方法?",
        r".*问题.*": "有什么问题?",
        r".*功能.*": "有什么功能?",
        r".*特点.*": "有什么特点?",
        r".*优势.*": "有什么优势?",
        r".*用途.*": "有什么用途?",
    }
    
    def __init__(self):
        logger.info("初始化基于规则的问题生成器")
    
    def generate(self, document: str, num_queries: int = 3) -> List[str]:
        """
        基于规则生成问题
        
        Args:
            document: 输入文档
            num_queries: 生成问题数量
        
        Returns:
            生成的问题列表
        """
        queries = []
        
        # 方法1: 基于关键词匹配
        for pattern, question in self.KEYWORD_PATTERNS.items():
            if re.search(pattern, document):
                if question not in queries:
                    queries.append(question)
        
        # 方法2: 从文档中提取名词短语，生成问题
        # 简单分词
        words = self._tokenize(document)
        
        # 提取关键实体（简单的名词识别）
        entities = self._extract_entities(words)
        
        # 使用模板生成问题
        for entity in entities[:3]:
            for keyword, template in self.QUESTION_TEMPLATES[:3]:
                query = template.format(entity)
                if query not in queries:
                    queries.append(query)
        
        # 方法3: 基于文档首句生成
        first_sentence = self._get_first_sentence(document)
        if first_sentence:
            queries.append(f"关于\"{first_sentence[:20]}...\"有什么详细信息?")
        
        # 限制数量并返回
        return queries[:num_queries]
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 去除标点符号，分割成词
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _extract_entities(self, words: List[str]) -> List[str]:
        """简单提取实体（名词）"""
        # 常见名词后缀
        noun_suffixes = ['的', '是', '在', '有', '和', '与', '或', '了']
        
        entities = []
        for i, word in enumerate(words):
            # 提取2-4字的中文词组
            if 2 <= len(word) <= 4 and word not in noun_suffixes:
                entities.append(word)
        
        return list(set(entities))
    
    def _get_first_sentence(self, text: str) -> str:
        """获取第一句话"""
        # 按句号、问号、感叹号分割
        sentences = re.split(r'[。？！]', text)
        return sentences[0] if sentences else ""


class GPT2BasedGenerator:
    """基于GPT-2的问题生成器"""
    
    # 生成问题的Prompt模板
    PROMPT_TEMPLATES = [
        "文档内容：{document}\n根据以上内容，可能的查询问题是：",
        "内容：{document}\n用户可能会问：",
        "{document}\n请生成三个相关问题：",
    ]
    
    # 常见问题前缀
    QUESTION_PREFIXES = [
        "什么是",
        "如何",
        "为什么",
        "怎么",
        "哪有",
        "谁",
        "多少",
        "什么时候",
        "为什么",
        "是否",
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化GPT-2生成器
        
        Args:
            model_path: 模型路径，默认使用uer/gpt2-chinese-cluecorpussmall
            device: 推理设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"初始化GPT-2生成器，设备: {self.device}")
        
        # 默认模型路径
        if model_path is None:
            model_path = "uer/gpt2-chinese-cluecorpussmall"
        
        logger.info(f"加载模型: {model_path}")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("模型加载成功")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}，将使用模拟模式")
            self.model = None
            self.tokenizer = None
    
    def generate(self, document: str, num_queries: int = 3) -> List[str]:
        """
        基于GPT-2生成问题
        
        Args:
            document: 输入文档
            num_queries: 生成问题数量
        
        Returns:
            生成的问题列表
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("模型未加载，使用默认问题")
            return self._generate_fallback_queries(document, num_queries)
        
        queries = []
        
        # 选择prompt模板
        prompt_template = self.PROMPT_TEMPLATES[0]
        prompt = prompt_template.format(document=document[:200])  # 限制长度
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # 生成参数
            output_ids = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=num_queries,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # 解码生成的文本
            for output in output_ids:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # 提取生成的问题部分
                questions = self._extract_questions(generated_text, prompt)
                queries.extend(questions)
        
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return self._generate_fallback_queries(document, num_queries)
        
        # 去重并限制数量
        unique_queries = list(dict.fromkeys(queries))
        return unique_queries[:num_queries]
    
    def _extract_questions(self, generated_text: str, prompt: str) -> List[str]:
        """从生成的文本中提取问题"""
        questions = []
        
        # 获取生成的部分（去除prompt）
        if prompt in generated_text:
            generated_part = generated_text[len(prompt):]
        else:
            generated_part = generated_text
        
        # 按换行或标点分割
        lines = re.split(r'[\n。；!?]', generated_part)
        
        for line in lines:
            line = line.strip()
            # 检查是否包含问号结尾
            if '?' in line or '？' in line:
                questions.append(line)
            # 检查是否以疑问词开头
            elif any(line.startswith(prefix) for prefix in self.QUESTION_PREFIXES):
                questions.append(line + "?")
        
        return questions
    
    def _generate_fallback_queries(self, document: str, num_queries: int) -> List[str]:
        """生成备用问题（当模型不可用时）"""
        queries = [
            f"关于\"{document[:15]}...\"的详细信息是什么?",
            f"\"{document[:10]}...\"有什么特点?",
            f"\"{document[:10]}...\"如何使用?",
        ]
        return queries[:num_queries]


class Doc2QueryModel:
    """
    doc2query 模型推理类
    
    支持两种模式：
    - rule-based: 基于规则的问题生成（快速，无需模型）
    - model-based: 基于GPT-2的问题生成（需要微调模型）
    """
    
    def __init__(self, model_path: str = None, device: str = "cuda", mode: str = "rule-based"):
        """
        初始化 doc2query 模型
        
        Args:
            model_path: 模型路径，默认使用uer/gpt2-chinese-cluecorpussmall
            device: 推理设备，"cuda" 或 "cpu"
            mode: 生成模式，"rule-based" 或 "model-based"
        """
        logger.info(f"初始化 Doc2QueryModel，模式: {mode}，设备: {device}")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mode = mode
        
        if mode == "rule-based":
            self.generator = RuleBasedGenerator()
            logger.info("使用基于规则的生成器")
        elif mode == "model-based":
            self.generator = GPT2BasedGenerator(model_path, self.device)
            logger.info("使用基于GPT-2的生成器")
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
    
    def generate(self, document: str, num_queries: int = 3) -> List[str]:
        """
        为文档生成潜在查询
        
        Args:
            document: 输入文档文本
            num_queries: 生成查询数量，默认3个
        
        Returns:
            生成的查询列表
        """
        logger.info(f"开始生成查询，文档长度: {len(document)}")
        
        queries = self.generator.generate(document, num_queries)
        
        logger.info(f"生成完成，共 {len(queries)} 个查询")
        for i, query in enumerate(queries):
            logger.debug(f"  查询 {i+1}: {query}")
        
        return queries
    
    def batch_generate(self, documents: List[str], num_queries: int = 3) -> List[List[str]]:
        """
        批量生成查询
        
        Args:
            documents: 文档列表
            num_queries: 每个文档生成的查询数量
        
        Returns:
            每条文档对应的查询列表
        """
        logger.info(f"开始批量生成查询，文档数量: {len(documents)}")
        
        results = []
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}")
            queries = self.generate(doc, num_queries)
            results.append(queries)
        
        logger.info("批量生成完成")
        return results
    
    def generate_with_format(self, document: str, comment_id: str, num_queries: int = 3) -> dict:
        """
        生成查询并返回标准格式
        
        Args:
            document: 输入文档文本
            comment_id: 评论/文档ID
            num_queries: 生成查询数量
        
        Returns:
            标准格式的字典，包含 comment_id 和 generated_queries
        """
        queries = self.generate(document, num_queries)
        
        return {
            "comment_id": comment_id,
            "generated_queries": queries
        }
    
    def batch_generate_with_format(
        self, 
        documents: List[str], 
        comment_ids: List[str] = None, 
        num_queries: int = 3
    ) -> List[dict]:
        """
        批量生成查询并返回标准格式
        
        Args:
            documents: 文档列表
            comment_ids: 评论/文档ID列表，如果为None则使用索引
            num_queries: 每个文档生成的查询数量
        
        Returns:
            标准格式的字典列表
        """
        if comment_ids is None:
            comment_ids = [f"doc_{i}" for i in range(len(documents))]
        
        results = []
        for doc, cid in zip(documents, comment_ids):
            result = self.generate_with_format(doc, cid, num_queries)
            results.append(result)
        
        return results
    
    @staticmethod
    def save_results(results: List[dict], output_path: str):
        """
        保存结果到JSON文件
        
        Args:
            results: 结果列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_path}")


def main():
    """演示用法"""
    # 示例文档
    sample_doc = """
    人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的机器。
    机器学习是人工智能的一个子领域，使用统计技术使计算机系统能够从数据中学习。
    深度学习是机器学习的一个分支，使用多层神经网络进行特征提取和模式识别。
    """
    
    print("=" * 60)
    print("Doc2Query 模型推理演示")
    print("=" * 60)
    
    # 模式1：基于规则的生成
    print("\n【模式A：基于规则的生成】")
    model_rule = Doc2QueryModel(mode="rule-based")
    queries_rule = model_rule.generate(sample_doc, num_queries=3)
    print(f"生成的查询: {queries_rule}")
    
    # 模式2：基于GPT-2的生成
    print("\n【模式B：基于GPT-2的生成】")
    try:
        model_gpt2 = Doc2QueryModel(mode="model-based", device="cpu")
        queries_gpt2 = model_gpt2.generate(sample_doc, num_queries=3)
        print(f"生成的查询: {queries_gpt2}")
    except Exception as e:
        print(f"GPT-2模式初始化失败: {e}")
    
    # 批量生成
    print("\n【批量生成示例】")
    docs = [
        "第一个文档内容，关于机器学习。",
        "第二个文档内容，关于深度学习。",
        "第三个文档内容，关于自然语言处理。"
    ]
    batch_results = model_rule.batch_generate(docs, num_queries=2)
    for i, queries in enumerate(batch_results):
        print(f"文档{i+1}: {queries}")
    
    # 标准格式输出
    print("\n【标准格式输出】")
    formatted_results = model_rule.batch_generate_with_format(
        docs, 
        comment_ids=["c1", "c2", "c3"],
        num_queries=3
    )
    print(json.dumps(formatted_results, ensure_ascii=False, indent=2))
    
    # 保存结果
    output_path = "doc2query_results.json"
    model_rule.save_results(formatted_results, output_path)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
