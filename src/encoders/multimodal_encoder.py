import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from pytube import YouTube
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
        """使用CLIP模型将文本编码为向量表示"""
        try:
            # 使用CLIP处理器处理文本
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
            
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
    
    def encode_youtube_video(self, youtube_url, temp_dir="./temp_videos"):
        """从YouTube URL下载视频并编码"""
        try:
            os.makedirs(temp_dir, exist_ok=True)
            yt = YouTube(youtube_url)
            video_path = yt.streams.filter(progressive=True, file_extension='mp4').first().download(temp_dir)
            
            # 提取视频标题和描述作为元数据
            metadata = {
                "title": yt.title,
                "description": yt.description,
                "author": yt.author,
                "url": youtube_url
            }
            
            # 编码视频
            embedding = self.encode_video(video_path)
            
            # 可选：删除临时文件
            os.remove(video_path)
            
            return embedding, metadata
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            return None, {}
    
    def generate_text_description(self, image_path, llm=None):
        """透過CLIP產生圖片內容描述"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 預定義一些常見的視覺概念
            candidate_labels = [
                "a photograph of a person",
                "a photograph of an animal",
                "a photograph of a landscape",
                "a photograph of food",
                "a photograph of a building",
                "a photograph of a vehicle",
                "a photograph of nature",
                "a photograph of technology",
                "a photograph of art",
                "a photograph of a group of people"
            ]
            
            # 處理圖像
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # 處理文本
            text_inputs = self.clip_processor(text=candidate_labels, return_tensors="pt", padding=True)
            
            # 計算特徵
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                text_features = self.clip_model.get_text_features(**text_inputs)
            
            # 計算相似度
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 獲取前3個最相似的標籤
            values, indices = similarity[0].topk(3)
            top_predictions = [(candidate_labels[i], values[idx].item()) for idx, i in enumerate(indices)]
            
            # 構造描述
            description = f"This image appears to be {top_predictions[0][0]} (confidence: {top_predictions[0][1]:.2f})"
            if top_predictions[1][1] > 0.15:  # 只有當第二個類別置信度足夠高時才添加
                description += f", and may also contain {top_predictions[1][0].replace('a photograph of ', '')}"
            
            # 如果提供了LLM，使用它來生成更豐富的描述
            if llm is not None:
                try:
                    # 獲取前3個類別作為提示
                    categories_text = ", ".join([label.replace('a photograph of ', '') for label, _ in top_predictions[:3]])
                    prompt = f"Based on image classification results suggesting this image contains: {categories_text}, generate a detailed description of what might be in this image."
                    response = llm.invoke(prompt)
                    if response and len(response) > 20:  # 确保LLM返回了有意义的描述
                        return response
                except Exception as e:
                    print(f"LLM description generation failed: {e}")
                    # 繼續使用基本描述
            
            return description
        except Exception as e:
            print(f"Error generating image description with CLIP: {e}")
            return "No description available"