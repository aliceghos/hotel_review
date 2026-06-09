"""
图片增强模块
支持超分辨率和图片描述生成
"""

import os
import io
import base64
import logging
from typing import Optional, Callable, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    "super_resolution": {
        "enabled": True,
        "model": "RealESRGAN_x4plus",
        "scale": 4,
        "tile_size": 200
    },
    "description": {
        "model": "qwen_vl",
        "api_key": None,
        "prompt": "请详细描述这张酒店图片的内容，包括房间设施、装修风格、色彩等细节"
    },
    "vectorization": {
        "model": "clip",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}


class ImageEnhancer:
    """图片增强处理器
    
    支持超分辨率放大和图片描述生成功能
    """
    
    def __init__(self, config: dict = None):
        """
        初始化图片增强器
        
        Args:
            config: 配置字典，包含:
                - use_super_resolution: 是否使用超分辨率
                - super_resolution_model: 超分辨率模型名称
                - description_model: 描述生成模型类型
                - device: 推理设备
        """
        self.config = self._merge_config(config)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型实例
        self._sr_model = None
        self._description_model = None
        self._clip_model = None
        self._clip_processor = None
        
        # 初始化各模块
        self._init_super_resolution()
        self._init_description_model()
        self._init_clip_model()
    
    def _merge_config(self, config: dict) -> dict:
        """合并用户配置与默认配置"""
        merged = DEFAULT_CONFIG.copy()
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in merged:
                    merged[key].update(value)
                else:
                    merged[key] = value
        return merged
    
    def _init_super_resolution(self):
        """初始化超分辨率模型"""
        if not self.config.get("super_resolution", {}).get("enabled", True):
            logger.info("超分辨率功能已禁用")
            return
        
        model_name = self.config.get("super_resolution", {}).get("model", "RealESRGAN_x4plus")
        
        try:
            if model_name == "RealESRGAN_x4plus":
                self._init_realesrgan()
            elif model_name == "swinir":
                self._init_swinir()
            else:
                logger.warning(f"不支持的超分辨率模型: {model_name}，将使用原图")
        except Exception as e:
            logger.warning(f"超分辨率模型初始化失败: {e}，将使用原图")
            self._sr_model = None
    
    def _init_realesrgan(self):
        """初始化 Real-ESRGAN 模型"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            scale = self.config.get("super_resolution", {}).get("scale", 4)
            tile_size = self.config.get("super_resolution", {}).get("tile_size", 200)
            
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale
            )
            
            model_path = os.path.join(
                os.path.dirname(__file__), 
                "models", 
                "RealESRGAN_x4plus.pth"
            )
            
            # 如果本地没有模型文件，尝试从网络下载
            if not os.path.exists(model_path):
                logger.info("正在下载 Real-ESRGAN 模型...")
                model_path = "RealESRGAN_x4plus"
            
            self._sr_model = RealESRGANer(
                scale=scale,
                tile_size=tile_size,
                tile_pad=10,
                pre_pad=0,
                model=model,
                model_path=model_path,
                device=self.device,
                half=(self.device == "cuda")
            )
            logger.info("Real-ESRGAN 模型加载成功")
            
        except ImportError as e:
            logger.warning(f"Real-ESRGAN 依赖未安装: {e}")
            self._sr_model = None
        except Exception as e:
            logger.warning(f"Real-ESRGAN 模型初始化失败: {e}")
            self._sr_model = None
    
    def _init_swinir(self):
        """初始化 SwinIR 模型"""
        try:
            from basicsr.archs.srformer_arch import SwinIR
            
            scale = self.config.get("super_resolution", {}).get("scale", 4)
            
            model = SwinIR(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=96,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2.,
                upside_down=False
            )
            
            self._sr_model = model
            self._sr_model.eval()
            self._sr_model.to(self.device)
            logger.info("SwinIR 模型加载成功")
            
        except Exception as e:
            logger.warning(f"SwinIR 模型初始化失败: {e}")
            self._sr_model = None
    
    def _init_description_model(self):
        """初始化图片描述生成模型"""
        model_type = self.config.get("description", {}).get("model", "qwen_vl")
        
        try:
            if model_type == "blip2":
                self._init_blip2()
            elif model_type == "llava":
                self._init_llava()
            elif model_type == "qwen_vl":
                # Qwen-VL 通过API调用，无需在此初始化
                logger.info("Qwen-VL 将通过API调用")
            else:
                logger.warning(f"不支持的描述模型: {model_type}")
        except Exception as e:
            logger.warning(f"描述模型初始化失败: {e}")
            self._description_model = None
    
    def _init_blip2(self):
        """初始化 BLIP-2 模型"""
        try:
            from transformers import Blip2Processor, Blip2Model
            
            model_name = "Salesforce/blip2-opt-2.7b"
            self._blip_processor = Blip2Processor.from_pretrained(model_name)
            self._blip_model = Blip2Model.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self._blip_model.to(self.device)
            self._blip_model.eval()
            self._description_model = "blip2"
            logger.info("BLIP-2 模型加载成功")
            
        except Exception as e:
            logger.warning(f"BLIP-2 模型初始化失败: {e}")
            self._description_model = None
    
    def _init_llava(self):
        """初始化 LLaVA 模型"""
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN
            
            model_path = self.config.get("description", {}).get("model_path", "liuhaotian/llava-v1.5-7b")
            
            self._llava_tokenizer, self._llava_model, self._llava_image_processor, _ = \
                load_pretrained_model(model_path)
            
            self._description_model = "llava"
            logger.info("LLaVA 模型加载成功")
            
        except Exception as e:
            logger.warning(f"LLaVA 模型初始化失败: {e}")
            self._description_model = None
    
    def _init_clip_model(self):
        """初始化 CLIP 模型用于向量化"""
        model_name = self.config.get("vectorization", {}).get("model", "clip")
        
        try:
            if model_name == "clip":
                from transformers import CLIPProcessor, CLIPModel
                
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model.to(self.device)
                self._clip_model.eval()
                logger.info("CLIP 模型加载成功")
            else:
                logger.warning(f"不支持的向量化模型: {model_name}")
                
        except Exception as e:
            logger.warning(f"CLIP 模型初始化失败: {e}")
            self._clip_model = None
    
    def _apply_super_resolution(self, image: Image.Image) -> Image.Image:
        """
        应用超分辨率模型
        
        Args:
            image: 输入图片 (PIL.Image)
            
        Returns:
            超分辨率处理后的图片
        """
        if self._sr_model is None:
            logger.info("超分辨率模型不可用，使用原图")
            return image
        
        # 确保图片是 RGB 格式
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 转换为 numpy 数组
        img = np.array(image)
        
        try:
            model_name = self.config.get("super_resolution", {}).get("model", "RealESRGAN_x4plus")
            
            if model_name == "RealESRGAN_x4plus":
                from realesrgan import RealESRGANer
                if isinstance(self._sr_model, RealESRGANer):
                    output, _ = self._sr_modelenhance_image(img)
                    return Image.fromarray(output)
            
            # SwinIR 或其他模型
            with torch.no_grad():
                # 转换为 tensor
                from basicsr.utils import img2tensor
                img_tensor = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # 执行推理
                output = self._sr_model(img_tensor)
                
                # 转换回图片
                output = output.squeeze(0).cpu().numpy()
                output = (output * 255.0).clip(0, 255).astype(np.uint8)
                output = np.transpose(output, (1, 2, 0))
                
                if output.shape[2] == 3:
                    output = output[:, :, ::-1]  # RGB to BGR
                
                return Image.fromarray(output)
                
        except Exception as e:
            logger.warning(f"超分辨率处理失败: {e}，使用原图")
            return image
    
    def _generate_description(self, image: Image.Image, prompt: str = None) -> str:
        """
        生成图片描述
        
        Args:
            image: 输入图片 (PIL.Image)
            prompt: 可选的提示词
            
        Returns:
            图片描述文本
        """
        model_type = self.config.get("description", {}).get("model", "qwen_vl")
        
        if model_type == "qwen_vl":
            return self._generate_description_qwen(image, prompt)
        elif model_type == "llava":
            return self._generate_description_llava(image, prompt)
        elif model_type == "blip2":
            return self._generate_description_blip2(image, prompt)
        else:
            logger.warning(f"不支持的描述模型: {model_type}")
            return ""
    
    def _generate_description_qwen(self, image: Image.Image, prompt: str = None) -> str:
        """使用 Qwen-VL API 生成描述"""
        api_key = self.config.get("description", {}).get("api_key")
        
        if not api_key:
            logger.warning("Qwen-VL API密钥未设置，尝试使用本地模型")
            return self._generate_description_blip2(image, prompt)
        
        try:
            import dashscope
            from dashscope import MultiModalConversation
            
            dashscope.api_key = api_key
            
            # 将图片转换为 base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            user_prompt = prompt or self.config.get("description", {}).get(
                "prompt", 
                "请详细描述这张图片的内容"
            )
            
            messages = [{
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{img_base64}"},
                    {"text": user_prompt}
                ]
            }]
            
            response = MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=messages
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content[0]["text"]
            else:
                logger.warning(f"Qwen-VL API 调用失败: {response.message}")
                return self._generate_description_blip2(image, prompt)
                
        except Exception as e:
            logger.warning(f"Qwen-VL API 调用失败: {e}")
            return self._generate_description_blip2(image, prompt)
    
    def _generate_description_llava(self, image: Image.Image, prompt: str = None) -> str:
        """使用 LLaVA 生成描述"""
        if self._description_model != "llava":
            logger.warning("LLaVA 模型未初始化")
            return ""
        
        try:
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates
            
            user_prompt = prompt or self.config.get("description", {}).get(
                "prompt",
                "请详细描述这张图片的内容"
            )
            
            image_size = 336
            images_tensor = process_images([image], image_size, self._llava_image_processor)
            
            if images_tensor is not None:
                images_tensor = images_tensor.to(self.device, dtype=torch.float16)
            
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], IMAGE_TOKEN + "\n" + user_prompt)
            conv.append_message(conv.roles[1], None)
            
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self._llava_tokenizer, IMAGE_TOKEN, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_ids = self._llava_model.generate(
                    input_ids,
                    images=images_tensor,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            output = self._llava_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output.strip()
            
        except Exception as e:
            logger.warning(f"LLaVA 描述生成失败: {e}")
            return self._generate_description_blip2(image, prompt)
    
    def _generate_description_blip2(self, image: Image.Image, prompt: str = None) -> str:
        """使用 BLIP-2 生成描述（本地模型，降级方案）"""
        if self._description_model != "blip2" or not hasattr(self, '_blip_model'):
            logger.info("尝试初始化 BLIP-2 作为降级方案")
            self._init_blip2()
            
            if self._description_model != "blip2":
                logger.warning("BLIP-2 模型不可用，返回空描述")
                return ""
        
        try:
            user_prompt = prompt or self.config.get("description", {}).get(
                "prompt",
                "请详细描述这张图片"
            )
            
            inputs = self._blip_processor(
                images=image,
                text=user_prompt,
                return_tensors="pt"
            ).to(self.device, dtype=torch.float16)
            
            with torch.no_grad():
                outputs = self._blip_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            description = self._blip_processor.decode(outputs[0], skip_special_tokens=True)
            return description.strip()
            
        except Exception as e:
            logger.warning(f"BLIP-2 描述生成失败: {e}")
            return ""
    
    def _encode_image_clip(self, image: Image.Image) -> Optional[list]:
        """使用 CLIP 编码图片"""
        if self._clip_model is None or self._clip_processor is None:
            logger.info("CLIP 模型不可用，跳过向量化")
            return None
        
        try:
            inputs = self._clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self._clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().tolist()[0]
            
        except Exception as e:
            logger.warning(f"CLIP 图片编码失败: {e}")
            return None
    
    def _encode_text_clip(self, text: str) -> Optional[list]:
        """使用 CLIP 编码文本"""
        if self._clip_model is None or self._clip_processor is None:
            return None
        
        try:
            inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self._clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().tolist()[0]
            
        except Exception as e:
            logger.warning(f"CLIP 文本编码失败: {e}")
            return None
    
    def enhance_image(
        self, 
        image_path: str, 
        output_dir: str = None,
        progress_callback: Callable[[str, int], None] = None
    ) -> dict:
        """
        增强单张图片
        
        Args:
            image_path: 输入图片路径
            output_dir: 输出目录，None则在原图目录生成enhanced_前缀的文件
            progress_callback: 进度回调函数，接收 (阶段名称, 进度百分比)
            
        Returns:
            {
                "original_path": str,
                "enhanced_path": str,  # 超分辨率后的图片路径
                "description": str,    # 图片描述
                "image_vector": list,  # CLIP图像向量
                "description_vector": list  # 描述文本向量
            }
        """
        result = {
            "original_path": image_path,
            "enhanced_path": None,
            "description": "",
            "image_vector": None,
            "description_vector": None,
            "error": None
        }
        
        try:
            # 1. 加载图片
            if progress_callback:
                progress_callback("loading", 0)
            
            image = Image.open(image_path)
            
            # 确保 RGB 格式
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            original_size = image.size
            logger.info(f"原始图片尺寸: {original_size}")
            
            # 2. 超分辨率处理
            if progress_callback:
                progress_callback("super_resolution", 20)
            
            if self._sr_model is not None and self.config.get("super_resolution", {}).get("enabled", True):
                image = self._apply_super_resolution(image)
                logger.info(f"超分辨率后尺寸: {image.size}")
            
            # 3. 保存增强后的图片
            if progress_callback:
                progress_callback("saving", 50)
            
            if output_dir is None:
                output_dir = os.path.dirname(image_path)
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            original_filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(original_filename)[0]
            ext = os.path.splitext(original_filename)[1] or ".png"
            enhanced_filename = f"enhanced_{name_without_ext}.png"
            enhanced_path = os.path.join(output_dir, enhanced_filename)
            
            image.save(enhanced_path, "PNG")
            result["enhanced_path"] = enhanced_path
            
            # 4. 生成描述
            if progress_callback:
                progress_callback("description", 70)
            
            result["description"] = self._generate_description(image)
            logger.info(f"图片描述: {result['description'][:100]}...")
            
            # 5. 向量化
            if progress_callback:
                progress_callback("vectorization", 90)
            
            result["image_vector"] = self._encode_image_clip(image)
            
            if result["description"]:
                result["description_vector"] = self._encode_text_clip(result["description"])
            
            if progress_callback:
                progress_callback("complete", 100)
            
        except Exception as e:
            error_msg = f"图片增强失败: {e}"
            logger.error(error_msg)
            result["error"] = error_msg
        
        return result
    
    def batch_enhance(
        self,
        image_paths: list[str],
        batch_size: int = 8,
        output_dir: str = None,
        progress_callback: Callable[[int, int, dict], None] = None
    ) -> list[dict]:
        """
        批量增强图片
        
        Args:
            image_paths: 图片路径列表
            batch_size: 每批处理的图片数量
            output_dir: 输出目录
            progress_callback: 进度回调函数，接收 (已完成数量, 总数, 当前图片结果)
            
        Returns:
            增强结果列表
        """
        results = []
        total = len(image_paths)
        
        # 分批处理
        for i in range(0, total, batch_size):
            batch = image_paths[i:i + batch_size]
            
            for j, image_path in enumerate(batch):
                logger.info(f"处理图片 {i + j + 1}/{total}: {image_path}")
                
                result = self.enhance_image(
                    image_path,
                    output_dir=output_dir,
                    progress_callback=lambda stage, prog: logger.debug(f"{stage}: {prog}%")
                )
                
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + j + 1, total, result)
        
        return results
    
    async def enhance_image_async(
        self,
        image_path: str,
        output_dir: str = None,
        progress_callback: Callable[[str, int], None] = None
    ) -> dict:
        """
        异步增强单张图片
        
        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            progress_callback: 进度回调函数
            
        Returns:
            增强结果字典
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.enhance_image,
            image_path,
            output_dir,
            progress_callback
        )
    
    async def batch_enhance_async(
        self,
        image_paths: list[str],
        batch_size: int = 8,
        output_dir: str = None,
        progress_callback: Callable[[int, int, dict], None] = None
    ) -> list[dict]:
        """
        异步批量增强图片
        
        Args:
            image_paths: 图片路径列表
            batch_size: 每批处理的图片数量
            output_dir: 输出目录
            progress_callback: 进度回调函数
            
        Returns:
            增强结果列表
        """
        tasks = []
        for image_path in image_paths:
            task = self.enhance_image_async(
                image_path,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            tasks.append(task)
        
        results = []
        for i, task in enumerate(as_completed(tasks)):
            result = await task
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(image_paths), result)
        
        return results
    
    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        return {
            "super_resolution": {
                "model": self.config.get("super_resolution", {}).get("model"),
                "enabled": self.config.get("super_resolution", {}).get("enabled", True),
                "available": self._sr_model is not None
            },
            "description": {
                "model": self.config.get("description", {}).get("model"),
                "available": self._description_model is not None or self.config.get("description", {}).get("api_key")
            },
            "clip": {
                "available": self._clip_model is not None
            },
            "device": self.device
        }
    
    def release_memory(self):
        """释放模型内存"""
        if self._sr_model is not None:
            del self._sr_model
            self._sr_model = None
        
        if self._clip_model is not None:
            del self._clip_model
            self._clip_model = None
        
        if hasattr(self, '_blip_model') and self._blip_model is not None:
            del self._blip_model
            self._blip_model = None
        
        if hasattr(self, '_llava_model') and self._llava_model is not None:
            del self._llava_model
            self._llava_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("模型内存已释放")
    
    def __del__(self):
        """析构函数"""
        self.release_memory()


def main():
    """演示用法"""
    # 创建图片增强器
    enhancer = ImageEnhancer()
    
    # 获取模型信息
    print("模型信息:", enhancer.get_model_info())
    
    # 单张图片增强
    # result = enhancer.enhance_image("path/to/image.jpg")
    # print(result)
    
    # 批量增强
    # results = enhancer.batch_enhance(
    #     ["path/to/image1.jpg", "path/to/image2.jpg"],
    #     batch_size=4
    # )
    
    # 释放内存
    enhancer.release_memory()


if __name__ == "__main__":
    main()
