"""
多模态向量构建模块

支持CLIP图像向量、描述向量和图文联合向量的构建
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import pickle
import hashlib

# 图像处理
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 深度学习框架
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# CLIP模型
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Transformer模型
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPVisionModel, CLIPTextModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# 支持的模型配置
SUPPORTED_MODELS = {
    "clip": {
        "name": "CLIP ViT-L/14",
        "vector_dim": 768,
        "pretrained": "openai/clip-vit-large-patch14"
    },
    "vit": {
        "name": "Vision Transformer",
        "vector_dim": 768,
        "pretrained": "google/vit-base-patch16-224"
    },
    "chinese_clip": {
        "name": "Chinese-CLIP",
        "vector_dim": 512,
        "pretrained": "OFA-Sys/chinese-clip-vit-base-patch16"
    }
}


@dataclass
class MultimodalVectors:
    """多模态向量容器"""
    image_vector: np.ndarray
    text_vector: np.ndarray
    cross_modal_vector: np.ndarray
    
    def to_dict(self) -> dict:
        return {
            "image_vector": self.image_vector,
            "text_vector": self.text_vector,
            "cross_modal_vector": self.cross_modal_vector
        }


class MultimodalVectorizer:
    """多模态向量构建器"""
    
    def __init__(self, config: dict = None):
        """
        初始化向量化器
        
        Args:
            config: 配置字典，包含以下可选字段：
                - model_type: str, 模型类型 ("clip", "vit", "chinese_clip")，默认 "clip"
                - device: str, 设备 ("cuda", "cpu")，默认自动选择
                - vector_dim: int, 向量维度，默认 768
                - batch_size: int, 批处理大小，默认 32
                - cross_modal_method: str, 图文联合方法 ("concat", "hadamard", "fusion")，默认 "fusion"
        """
        self.config = config or {}
        
        # 模型配置
        self.model_type = self.config.get("model_type", "clip")
        self.vector_dim = self.config.get("vector_dim", 768)
        self.batch_size = self.config.get("batch_size", 32)
        self.cross_modal_method = self.config.get("cross_modal_method", "fusion")
        
        # 设备选择
        if self.config.get("device"):
            self.device = self.config["device"]
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 模型和处理器
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._initialized = False
    
    def _initialize(self):
        """延迟初始化模型和处理器"""
        if self._initialized:
            return
        
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required but not installed")
        
        if self.model_type == "clip":
            self._initialize_clip()
        elif self.model_type == "chinese_clip":
            self._initialize_chinese_clip()
        elif self.model_type == "vit":
            self._initialize_vit()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self._initialized = True
    
    def _initialize_clip(self):
        """初始化OpenAI CLIP模型"""
        if TRANSFORMERS_AVAILABLE:
            # 使用transformers的CLIP模型
            self._vision_model = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            self._text_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.vector_dim = 768
        elif CLIP_AVAILABLE:
            # 使用clip包
            self._clip_model, self._preprocess = clip.load("ViT-L/14", device=self.device)
            self.vector_dim = 768
        else:
            raise RuntimeError(
                "Neither transformers nor openai-clip is installed. "
                "Please install with: pip install transformers or pip install openai-clip"
            )
    
    def _initialize_chinese_clip(self):
        """初始化Chinese-CLIP模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is required for chinese_clip")
        
        self._vision_model = AutoModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16"
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16"
        )
        self.vector_dim = 512
    
    def _initialize_vit(self):
        """初始化Vision Transformer模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is required for vit")
        
        self._vision_model = AutoModel.from_pretrained(
            "google/vit-base-patch16-224"
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.vector_dim = 768
    
    def _ensure_initialized(self):
        """确保模型已初始化"""
        if not self._initialized:
            self._initialize()
    
    @torch.no_grad()
    def encode_image(self, image: "Image.Image") -> np.ndarray:
        """
        使用CLIP编码图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            512维或768维图像向量
        """
        self._ensure_initialized()
        
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required")
        
        # 确保图像是RGB模式
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.model_type == "clip" and CLIP_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            # 使用openai-clip包
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            features = self._clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()
        else:
            # 使用transformers
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if hasattr(self, "_vision_model"):
                outputs = self._vision_model(**inputs)
                # 使用[CLS] token或平均池化
                if hasattr(outputs, "last_hidden_state"):
                    image_features = outputs.last_hidden_state[:, 0, :]
                else:
                    image_features = outputs.pooler_output
            else:
                raise RuntimeError("Vision model not initialized")
            
            # L2归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        使用CLIP编码文本
        
        Args:
            text: 文本字符串
            
        Returns:
            512维或768维文本向量
        """
        self._ensure_initialized()
        
        if self.model_type == "clip" and CLIP_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            # 使用openai-clip包
            text_input = clip.tokenize([text]).to(self.device)
            features = self._clip_model.encode_text(text_input)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()
        else:
            # 使用transformers
            inputs = self._tokenizer([text], padding=True, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if hasattr(self, "_text_model"):
                outputs = self._text_model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    text_features = outputs.last_hidden_state[:, 0, :]
                else:
                    text_features = outputs.pooler_output
            else:
                # 对于纯视觉模型，使用文本描述的简单处理
                raise RuntimeError("Text model not available")
            
            # L2归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().squeeze()
    
    def encode_image_description_pair(
        self, 
        image: "Image.Image", 
        description: str
    ) -> Dict[str, np.ndarray]:
        """
        编码图像-描述对
        
        Args:
            image: PIL图像对象
            description: 图像描述文本
            
        Returns:
            包含以下键的字典:
                - image_vector: 图像向量
                - text_vector: 文本向量
                - cross_modal_vector: 图文联合向量
        """
        image_vector = self.encode_image(image)
        text_vector = self.encode_text(description)
        cross_modal_vector = self.build_cross_modal_vector(image_vector, text_vector)
        
        return {
            "image_vector": image_vector,
            "text_vector": text_vector,
            "cross_modal_vector": cross_modal_vector
        }
    
    def build_cross_modal_vector(
        self, 
        image_vector: np.ndarray, 
        text_vector: np.ndarray
    ) -> np.ndarray:
        """
        构建图文联合向量
        
        提供了三种构建方法:
        1. 直接拼接 [image_vec; text_vec]
        2. 逐元素乘法 (Hadamard product)
        3. 拼接 + 非线性变换 (fusion)
        
        Args:
            image_vector: 图像向量
            text_vector: 文本向量
            
        Returns:
            联合向量
        """
        if self.cross_modal_method == "concat":
            # 方法1: 直接拼接
            return np.concatenate([image_vector, text_vector])
        
        elif self.cross_modal_method == "hadamard":
            # 方法2: 逐元素乘法
            return image_vector * text_vector
        
        elif self.cross_modal_method == "fusion":
            # 方法3: 拼接 + 非线性变换
            concat_vector = np.concatenate([image_vector, text_vector])
            
            # 简单的非线性变换 (使用tanh激活)
            # 在实际应用中可以使用更复杂的MLP
            fused = np.tanh(concat_vector)
            
            # 再次与原始向量结合
            cross_modal = np.concatenate([
                image_vector * text_vector,  # Hadamard
                fused  # 拼接后的非线性变换
            ])
            
            return cross_modal
        
        else:
            raise ValueError(f"Unknown cross-modal method: {self.cross_modal_method}")
    
    def batch_encode_images(
        self, 
        images: List["Image.Image"], 
        batch_size: int = None
    ) -> List[np.ndarray]:
        """
        批量编码图像
        
        Args:
            images: PIL图像列表
            batch_size: 批处理大小，默认使用初始化时的值
            
        Returns:
            图像向量列表
        """
        self._ensure_initialized()
        
        batch_size = batch_size or self.batch_size
        vectors = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            if self.model_type == "clip" and CLIP_AVAILABLE and not TRANSFORMERS_AVAILABLE:
                # 使用openai-clip包的批量处理
                batch_inputs = [self._preprocess(img) for img in batch]
                batch_tensor = torch.stack(batch_inputs).to(self.device)
                features = self._clip_model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.extend(features.cpu().numpy())
            else:
                # 使用transformers的批量处理
                for img in batch:
                    vec = self.encode_image(img)
                    vectors.append(vec)
        
        return vectors
    
    def batch_encode_descriptions(
        self, 
        descriptions: List[str], 
        batch_size: int = None
    ) -> List[np.ndarray]:
        """
        批量编码描述文本
        
        Args:
            descriptions: 描述文本列表
            batch_size: 批处理大小，默认使用初始化时的值
            
        Returns:
            文本向量列表
        """
        self._ensure_initialized()
        
        batch_size = batch_size or self.batch_size
        vectors = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            
            if self.model_type == "clip" and CLIP_AVAILABLE and not TRANSFORMERS_AVAILABLE:
                # 使用openai-clip包的批量处理
                text_inputs = clip.tokenize(batch).to(self.device)
                features = self._clip_model.encode_text(text_inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.extend(features.cpu().numpy())
            else:
                # 使用transformers的批量处理
                inputs = self._tokenizer(
                    batch, 
                    padding=True, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=77
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self._text_model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    text_features = outputs.last_hidden_state[:, 0, :]
                else:
                    text_features = outputs.pooler_output
                
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                vectors.extend(text_features.cpu().numpy())
        
        return vectors
    
    def build_multimodal_index(
        self, 
        comment_images: List[dict], 
        persist_dir: str
    ) -> dict:
        """
        构建多模态索引
        
        Args:
            comment_images: 评论图片列表，格式如下:
                [{
                    "comment_id": "xxx",
                    "images": [
                        {
                            "image_path": "path/to/image.jpg",
                            "description": "图片描述"
                        }
                    ]
                }]
            persist_dir: 持久化存储目录
            
        Returns:
            包含以下键的字典:
                - comment_ids: 评论ID列表
                - image_vectors: 图像向量列表
                - description_vectors: 描述向量列表
                - cross_modal_vectors: 图文联合向量列表
                - metadata: 额外元数据
        """
        self._ensure_initialized()
        
        # 创建输出目录
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        comment_ids = []
        image_vectors = []
        description_vectors = []
        cross_modal_vectors = []
        metadata = {
            "image_paths": [],
            "descriptions": [],
            "model_type": self.model_type,
            "vector_dim": self.vector_dim,
            "cross_modal_method": self.cross_modal_method,
            "cross_modal_dim": self._get_cross_modal_dim()
        }
        
        for comment in comment_images:
            comment_id = comment.get("comment_id", "")
            images = comment.get("images", [])
            
            for img_data in images:
                image_path = img_data.get("image_path", "")
                description = img_data.get("description", "")
                
                try:
                    # 加载图像
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                    else:
                        # 如果图像不存在，跳过
                        continue
                    
                    # 编码
                    vectors = self.encode_image_description_pair(image, description)
                    
                    comment_ids.append(comment_id)
                    image_vectors.append(vectors["image_vector"])
                    description_vectors.append(vectors["text_vector"])
                    cross_modal_vectors.append(vectors["cross_modal_vector"])
                    metadata["image_paths"].append(image_path)
                    metadata["descriptions"].append(description)
                    
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue
        
        # 转换为numpy数组
        result = {
            "comment_ids": comment_ids,
            "image_vectors": np.array(image_vectors) if image_vectors else np.array([]),
            "description_vectors": np.array(description_vectors) if description_vectors else np.array([]),
            "cross_modal_vectors": np.array(cross_modal_vectors) if cross_modal_vectors else np.array([]),
            "metadata": metadata
        }
        
        # 持久化存储
        self._persist_index(result, persist_path)
        
        return result
    
    def _get_cross_modal_dim(self) -> int:
        """获取图文联合向量的维度"""
        if self.cross_modal_method == "concat":
            return self.vector_dim * 2
        elif self.cross_modal_method == "hadamard":
            return self.vector_dim
        elif self.cross_modal_method == "fusion":
            return self.vector_dim * 3
        return self.vector_dim
    
    def _persist_index(self, index_data: dict, persist_path: Path):
        """持久化索引数据"""
        # 保存为主索引文件
        index_file = persist_path / "multimodal_index.pkl"
        with open(index_file, "wb") as f:
            pickle.dump(index_data, f)
        
        # 保存为JSON元数据
        metadata_file = persist_path / "index_metadata.json"
        json_metadata = {
            "comment_ids": index_data["comment_ids"],
            "image_count": len(index_data["image_vectors"]),
            "vector_dim": int(index_data["metadata"]["vector_dim"]),
            "cross_modal_dim": int(index_data["metadata"]["cross_modal_dim"]),
            "model_type": index_data["metadata"]["model_type"],
            "cross_modal_method": index_data["metadata"]["cross_modal_method"],
            "image_paths": index_data["metadata"]["image_paths"],
            "descriptions": index_data["metadata"]["descriptions"]
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(json_metadata, f, ensure_ascii=False, indent=2)
        
        # 保存向量为numpy格式
        if len(index_data["image_vectors"]) > 0:
            np.save(persist_path / "image_vectors.npy", index_data["image_vectors"])
            np.save(persist_path / "description_vectors.npy", index_data["description_vectors"])
            np.save(persist_path / "cross_modal_vectors.npy", index_data["cross_modal_vectors"])
        
        # 生成索引文件hash用于校验
        index_hash = hashlib.md5(
            str(index_data["comment_ids"]).encode()
        ).hexdigest()[:8]
        
        print(f"Index saved to {persist_path}")
        print(f"Total images indexed: {len(index_data['comment_ids'])}")
        print(f"Index hash: {index_hash}")
    
    @classmethod
    def load_index(cls, persist_dir: str) -> dict:
        """
        加载持久化的索引
        
        Args:
            persist_dir: 索引存储目录
            
        Returns:
            索引数据字典
        """
        persist_path = Path(persist_dir)
        index_file = persist_path / "multimodal_index.pkl"
        
        if index_file.exists():
            with open(index_file, "rb") as f:
                return pickle.load(f)
        
        raise FileNotFoundError(f"Index not found at {index_file}")
    
    def get_vector_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray, 
        method: str = "cosine"
    ) -> float:
        """
        计算两个向量的相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            method: 相似度计算方法 ("cosine", "dot")
            
        Returns:
            相似度分数
        """
        if method == "cosine":
            # 余弦相似度
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)
        elif method == "dot":
            # 点积
            return np.dot(vec1, vec2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        index_data: dict, 
        top_k: int = 5,
        vector_type: str = "cross_modal"
    ) -> List[Dict]:
        """
        在索引中搜索相似向量
        
        Args:
            query_vector: 查询向量
            index_data: 索引数据
            top_k: 返回前k个结果
            vector_type: 查询向量类型 ("image", "description", "cross_modal")
            
        Returns:
            相似结果列表，每项包含comment_id, image_path, description, score
        """
        if vector_type == "image":
            vectors = index_data["image_vectors"]
        elif vector_type == "description":
            vectors = index_data["description_vectors"]
        else:
            vectors = index_data["cross_modal_vectors"]
        
        # 计算相似度
        similarities = []
        for i, vec in enumerate(vectors):
            score = self.get_vector_similarity(query_vector, vec)
            similarities.append({
                "index": i,
                "comment_id": index_data["comment_ids"][i],
                "image_path": index_data["metadata"]["image_paths"][i],
                "description": index_data["metadata"]["descriptions"][i],
                "score": float(score)
            })
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]


# 异步版本的向量化器
class AsyncMultimodalVectorizer(MultimodalVectorizer):
    """异步多模态向量化器"""
    
    async def encode_image_async(self, image: "Image.Image") -> np.ndarray:
        """异步编码图像"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_image, image)
    
    async def encode_text_async(self, text: str) -> np.ndarray:
        """异步编码文本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_text, text)
    
    async def encode_image_description_pair_async(
        self, 
        image: "Image.Image", 
        description: str
    ) -> Dict[str, np.ndarray]:
        """异步编码图像-描述对"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.encode_image_description_pair, 
            image, 
            description
        )
    
    async def batch_encode_images_async(
        self, 
        images: List["Image.Image"], 
        batch_size: int = None
    ) -> List[np.ndarray]:
        """异步批量编码图像"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.batch_encode_images, 
            images, 
            batch_size
        )
    
    async def batch_encode_descriptions_async(
        self, 
        descriptions: List[str], 
        batch_size: int = None
    ) -> List[np.ndarray]:
        """异步批量编码描述"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.batch_encode_descriptions, 
            descriptions, 
            batch_size
        )


# ChromaDB兼容接口
class ChromaDBVectorizer(MultimodalVectorizer):
    """ChromaDB兼容的多模态向量化器"""
    
    def encode_image_for_chroma(self, image: "Image.Image") -> List[float]:
        """编码图像并返回ChromaDB兼容格式"""
        vector = self.encode_image(image)
        return vector.tolist()
    
    def encode_text_for_chroma(self, text: str) -> List[float]:
        """编码文本并返回ChromaDB兼容格式"""
        vector = self.encode_text(text)
        return vector.tolist()
    
    def encode_for_chroma(self, image: "Image.Image", description: str) -> dict:
        """编码图像-描述对并返回ChromaDB兼容格式"""
        vectors = self.encode_image_description_pair(image, description)
        return {
            "image_embedding": vectors["image_vector"].tolist(),
            "text_embedding": vectors["text_vector"].tolist(),
            "cross_modal_embedding": vectors["cross_modal_vector"].tolist()
        }


# DashVector兼容接口
class DashVectorVectorizer(MultimodalVectorizer):
    """DashVector兼容的多模态向量化器"""
    
    def encode_image_for_dash(self, image: "Image.Image") -> List[float]:
        """编码图像并返回DashVector兼容格式"""
        vector = self.encode_image(image)
        return vector.tolist()
    
    def encode_text_for_dash(self, text: str) -> List[float]:
        """编码文本并返回DashVector兼容格式"""
        vector = self.encode_text(text)
        return vector.tolist()
    
    def encode_for_dash(self, image: "Image.Image", description: str) -> dict:
        """编码图像-描述对并返回DashVector兼容格式"""
        vectors = self.encode_image_description_pair(image, description)
        return {
            "image_vector": vectors["image_vector"].tolist(),
            "text_vector": vectors["text_vector"].tolist(),
            "cross_modal_vector": vectors["cross_modal_vector"].tolist()
        }


if __name__ == "__main__":
    # 测试代码
    print("Supported models:", list(SUPPORTED_MODELS.keys()))
    print("Initializing MultimodalVectorizer...")
    
    # 创建向量化器实例
    config = {
        "model_type": "clip",
        "device": "cpu",
        "batch_size": 8
    }
    
    vectorizer = MultimodalVectorizer(config)
    print(f"Device: {vectorizer.device}")
    print(f"Vector dimension: {vectorizer.vector_dim}")
