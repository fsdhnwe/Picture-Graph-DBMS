import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import CLIPProcessor, CLIPModel

# 多模态编码器初始化
class MultiModalEncoder:
    def __init__(self):
        # 使用CLIP模型進行圖像和文本編碼
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()  # 設置為評估模式
        
        # 文本编码器 - 用於純文本的額外編碼
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def encode_image(self, image_path):
        """使用CLIP模型将图片编码为向量表示"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 使用CLIP处理器处理图像
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # 通过模型获取特征
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
            # 将特征压缩为一维向量并归一化
            image_embedding = image_features.squeeze().cpu().numpy()
            # 标准化向量
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            
            # CLIP图像特征的维度是512
            return image_embedding
        except Exception as e:
            print(f"Error encoding image with CLIP: {e}")
            return None
    
    def encode_text(self, text):
        """使用CLIP模型将文本編碼為向量表示"""
        try:
            # 使用CLIP处理器处理文本
            translate_text = self.translate_text(text)
            inputs = self.clip_processor(text=translate_text, return_tensors="pt", padding=True, truncation=True)

            # 通过模型获取特征
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                
            # 将特征压缩为一维向量并归一化
            text_embedding = text_features.squeeze().cpu().numpy()
            # 标准化向量
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            
            return text_embedding
        except Exception as e:
            print(f"Error encoding text with CLIP: {e}")
            return None
            
    def translate_text(self, text):
        """检测中文文本并翻译成英文
        
        Args:
            text (str): 输入文本，可以是中文或其他语言
            
        Returns:
            str: 如果输入是中文则返回英文翻译，否则返回原文
        """
        try:
            # 检测文本是否包含中文字符
            def contains_chinese(text):
                for char in text:
                    if '\u4e00' <= char <= '\u9fff':
                        return True
                return False
            
            # 如果不包含中文，直接返回原文
            if not contains_chinese(text):
                return text
                
            # 导入必要的模型和分词器
            import argostranslate.package, argostranslate.translate

            from_code = "zh"
            to_code = "en"

            # 更新語言包索引
            argostranslate.package.update_package_index()
            # 找出並安裝你想要的語言包 (e.g. 中文到英文)
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(lambda x: x.from_code == 'zh' and x.to_code == 'en', available_packages)
            )
            argostranslate.package.install_from_path(package_to_install.download())
            translated_text = argostranslate.translate.translate(text, from_code, to_code)
            print(f"Translated from Chinese to English: {text} -> {translated_text}")
            return translated_text
            
        except Exception as e:
            print(f"Error translating text: {e}")
            return text  # 出错时返回原文
    
    def encode_video(self, video_path, num_frames=8):
        """使用CLIP将视频编码为向量表示，抽取关键帧后编码"""
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算抽帧间隔
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            # 抽取关键帧并进行编码
            frame_features = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # 转换BGR格式为RGB并转为PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 使用CLIP处理器处理图像
                    inputs = self.clip_processor(images=pil_image, return_tensors="pt")
                    
                    # 通过模型获取特征
                    with torch.no_grad():
                        features = self.clip_model.get_image_features(**inputs)
                    
                    # 添加到特征列表
                    frame_features.append(features.squeeze().cpu().numpy())
            
            # 如果没有成功提取特征，返回None
            if not frame_features:
                return None
                
            # 通过平均所有帧的特征来获得视频的整体特征
            video_embedding = np.mean(frame_features, axis=0)
            # 标准化向量
            video_embedding = video_embedding / np.linalg.norm(video_embedding)
            return video_embedding
            
        except Exception as e:
            print(f"Error encoding video with CLIP: {e}")
            return None
    
    def generate_text_description(self, image_path):
        """使用BLIP模型生成圖片描述"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            from PIL import Image
            
            # 初始化BLIP模型和處理器
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # 讀取並處理圖片
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            print(image_path)
            
            # 生成描述
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
                description = processor.decode(out[0], skip_special_tokens=True)
            
            # 確保描述的首字母大寫
            description = description[0].upper() + description[1:]
            
            return description
            
        except Exception as e:
            print(f"Error generating image description with BLIP: {e}")
            return "No description available"