import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from pytube import YouTube
from langchain_huggingface import HuggingFaceEmbeddings

# 多模态编码器初始化
class MultiModalEncoder:
    def __init__(self):
        # 使用预训练的ResNet152进行图像编码
        self.image_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        # 移除最后的全连接层，只保留特征提取部分
        self.image_model = torch.nn.Sequential(*(list(self.image_model.children())[:-1]))
        self.image_model.eval()  # 设置为评估模式
        
        # 定义图像预处理流程
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 为视频特征提取初始化另一个ResNet模型
        self.video_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.video_model = torch.nn.Sequential(*(list(self.video_model.children())[:-1]))
        self.video_model.eval()
        
        # 文本编码器 - 使用本地Sentence Transformers
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def encode_image(self, image_path):
        """使用ResNet模型将图片编码为向量表示"""
        try:
            # 加载图像并应用预处理
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)  # 添加批处理维度
            
            # 通过模型获取特征
            with torch.no_grad():
                features = self.image_model(image_tensor)
                
            # 将特征压缩为一维向量
            image_embedding = features.squeeze().flatten().numpy()
            
            # ResNet152的特征维度为2048
            return image_embedding
        except Exception as e:
            print(f"Error encoding image with ResNet: {e}")
            return None
    
    def encode_video(self, video_path, num_frames=8):
        """使用ResNet将视频编码为向量表示，抽取关键帧后编码"""
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
                    
                    # 预处理并通过ResNet提取特征
                    image_tensor = self.image_transform(pil_image).unsqueeze(0)
                    with torch.no_grad():
                        features = self.video_model(image_tensor)
                    
                    # 添加到特征列表
                    frame_features.append(features.squeeze().flatten().numpy())
            
            # 如果没有成功提取特征，返回None
            if not frame_features:
                return None
                
            # 通过平均所有帧的特征来获得视频的整体特征
            video_embedding = np.mean(frame_features, axis=0)
            return video_embedding
            
        except Exception as e:
            print(f"Error encoding video with ResNet: {e}")
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
        """透過圖像分類產生圖片內容描述

            使用ResNet的ImageNet分類結果來產生描述
        """
        try:
            # 加載完整的ResNet模型（包含分類層）用於識別
            from torchvision.models import ResNet50_Weights
            classification_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            classification_model.eval()
            
            # 加載並預處理圖像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.image_transform(image).unsqueeze(0)
            
            # 取得預測
            with torch.no_grad():
                output = classification_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            # 獲取前5個預測結果
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            # 使用內置的ImageNet類別映射（在更新的torchvision中可用）
            try:
                from torchvision.models.resnet import _IMAGENET_CATEGORIES
                categories = [_IMAGENET_CATEGORIES[idx] for idx in top5_catid]
            except (ImportError, AttributeError):
                # 如果不能直接訪問類別名稱，使用更通用的描述
                categories = [f"category_{idx.item()}" for idx in top5_catid]
                
            confidences = [float(prob) for prob in top5_prob]
            
            # 構造描述
            description = f"This image appears to contain {categories[0]} (confidence: {confidences[0]:.2f})"
            if confidences[1] > 0.15:  # 只有當第二個類別置信度足夠高時才添加
                description += f", and may also contain {categories[1]}"
            
            # 如果提供了LLM，使用它來生成更豐富的描述
            if llm is not None:
                try:
                    # 獲取前5個類別作為提示
                    categories_text = ", ".join(categories[:3])
                    prompt = f"Based on image classification results suggesting this image contains: {categories_text}, generate a detailed description of what might be in this image."
                    response = llm.invoke(prompt)
                    if response and len(response) > 20:  # 确保LLM返回了有意义的描述
                        return response
                except Exception as e:
                    print(f"LLM description generation failed: {e}")
                    # 繼續使用基本描述
            
            return description
        except Exception as e:
            print(f"Error generating image description with ResNet: {e}")
            return "No description available"