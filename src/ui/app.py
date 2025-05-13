import os
import sys
import gradio as gr
import tempfile
from pathlib import Path
import shutil
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import io
from sklearn.manifold import TSNE
import base64
import pandas as pd
import tempfile
import uuid
import traceback

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 獲取 src 目錄
project_root = os.path.dirname(current_dir)  # 獲取項目根目錄
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 取得當前腳本(app.py)所在的目錄，然後向上移動一層到專案根目錄
project_root = Path(__file__).parent.parent.absolute()
# 構建 temp 資料夾的絕對路徑
temp_dir = project_root / "temp"
# 設定環境變數
os.environ['GRADIO_TEMP_DIR'] = str(temp_dir)
# 確保 temp 資料夾存在
temp_dir.mkdir(exist_ok=True)

# 使用絕對路徑導入
from src.database.neo4j_graph_rag import Neo4jGraphRAG
from src.config import NEO4J_DATABASE

class ImageDatabaseUI:
    def __init__(self):
        # 初始化資料庫連接
        self.db = Neo4jGraphRAG(use_local_llm=True)
        
        # 創建臨時目錄存放上傳圖片
        self.temp_dir = tempfile.mkdtemp()
        
        # 圖片儲存路徑
        self.image_dir = os.path.join(os.getcwd(), "test_images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 搜尋結果暫存
        self.current_search_results = []
        
        # 畫廊結果暫存
        self.current_gallery_images = []
        
        # 初始化 YOLO 模型用於自動標記
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")  # 加載小型 YOLOv8 模型
            self.has_yolo = True
            print("成功加載 YOLOv8 模型")
        except Exception as e:
            self.has_yolo = False
            print(f"無法加載 YOLOv8 模型: {e}")
    
    def auto_tag_image(self, image_path, max_tags=5):
        """使用 YOLOv8 自動標記圖片中的物件"""
        if not self.has_yolo:
            return []
        
        try:
            # 執行目標檢測
            results = self.yolo_model(image_path)
            
            # 獲取檢測到的類別名稱
            detected_objects = {}
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    cls_name = r.names[cls_id]
                    
                    # 只保留置信度高的檢測結果
                    if conf > 0.4:  # 設置置信度閾值
                        # 使用類別名稱作為鍵，存儲最高置信度
                        if cls_name not in detected_objects or detected_objects[cls_name] < conf:
                            detected_objects[cls_name] = conf
            
            # 按置信度排序並限制標籤數量
            sorted_objects = sorted(detected_objects.items(), key=lambda x: x[1], reverse=True)
            auto_tags = [cls_name for cls_name, _ in sorted_objects[:max_tags]]
            
            return auto_tags
        except Exception as e:
            print(f"自動標記圖片時發生錯誤: {e}")
            return []
    
    def upload_image(self, image, description=None, selected_tags=None, new_tags=None, use_auto_tagging=True):
        """上傳圖片到資料庫"""
        if image is None:
            return "請選擇圖片上傳", None, []
        
        # 保存圖片到測試圖片目錄
        timestamp = int(time.time())
        
        try:
            if isinstance(image, str):
                image_path = image
                extension = os.path.splitext(image_path)[1]
                save_path = os.path.join(self.image_dir, f"image_{timestamp}{extension}")
                shutil.copy(image_path, save_path)
            else:  # Gradio Image 對象 (可能是 PIL.Image 或其他格式)
                if hasattr(image, 'save'):  # PIL.Image 對象
                    extension = ".jpg"
                    save_path = os.path.join(self.image_dir, f"image_{timestamp}{extension}")
                    image.save(save_path)
                elif hasattr(image, 'astype'):  # NumPy 陣列
                    extension = ".jpg"
                    save_path = os.path.join(self.image_dir, f"image_{timestamp}{extension}")
                    pil_img = Image.fromarray(image.astype('uint8'))
                    pil_img.save(save_path)
                else:
                    return f"不支援的圖片格式: {type(image)}", None, []
            
            # 準備標籤
            all_tags = []
            
            # 添加自動標記的標籤
            if use_auto_tagging and self.has_yolo:
                auto_tags = self.auto_tag_image(save_path)
                all_tags.extend(auto_tags)
                print(f"自動標記標籤: {auto_tags}")
            
            # 添加選擇的現有標籤
            if selected_tags and isinstance(selected_tags, list):
                all_tags.extend(selected_tags)
            
            # 添加新輸入的標籤
            if new_tags:
                # 處理標籤字符串，支持逗號、分號或空格分隔
                if isinstance(new_tags, str):
                    # 替換標點符號為空格，然後分割
                    new_tags = new_tags.replace(',', ' ').replace(';', ' ')
                    tag_list = [tag.strip() for tag in new_tags.split() if tag.strip()]
                    all_tags.extend(tag_list)
            
            # 去除重複
            all_tags = list(set(all_tags))
            
            # 準備元數據
            metadata = {}
            if description:
                metadata["user_description"] = description
            
            # 添加到資料庫
            doc_id = self.db.add_image(save_path, metadata, all_tags)
            
            # 獲取所有標籤，包括數據庫中已有的
            current_tags = self.db.get_image_tags(doc_id)
            tag_names = [tag["name"] for tag in current_tags]
            
            return f"圖片上傳成功！ID: {doc_id}\n已添加標籤: {', '.join(all_tags)}", save_path, tag_names
        except Exception as e:
            return f"圖片上傳失敗: {str(e)}", None, []
    
    def search_images(self, query, min_similarity=0.65):
        """搜尋相似圖片"""
        if not query.strip():
            return "請輸入搜尋關鍵詞", [], None
        
        try:
            # 執行搜尋
            results = self.db.advanced_search(query, k=9, min_similarity=min_similarity)
            
            # 保存結果以便顯示詳情
            self.current_search_results = results
            
            # 提取圖片路徑
            image_paths = []
            for doc in results:
                path = doc.metadata.get("path")
                if path and os.path.exists(path):
                    image_paths.append((path, doc.metadata.get("score", 0)))
                else:
                    # 無效路徑，可能添加預設圖像
                    pass
            
            # 格式化結果訊息
            if image_paths:
                result_text = f"找到 {len(image_paths)} 個相關圖片"
                for i, (path, score) in enumerate(image_paths):
                    result_text += f"\n{i+1}. {os.path.basename(path)} (相似度: {score:.4f})"
            else:
                result_text = "沒有找到相關圖片"
                return result_text, [], None
            
            # 生成查詢詞與圖片的t-SNE可視化
            tsne_html = self._generate_search_tsne(query, [path for path, _ in image_paths])
            
            # 只返回路徑列表用於圖片展示
            return result_text, [path for path, _ in image_paths], tsne_html
        
        except Exception as e:
            return f"搜尋失敗: {str(e)}", [], None
            
    def _generate_search_tsne(self, query, image_paths):
        """為搜尋查詢和檢索到的圖片生成t-SNE可視化"""
        try:
            if not image_paths:
                return None
                
            # 收集嵌入向量
            query_embedding = self.db.encoder.encode_text(query)
            image_embeddings = []
            labels = []
            
            # 為每個圖片獲取嵌入
            for path in image_paths:
                if not os.path.exists(path):
                    continue
                
                # 獲取圖片嵌入
                img_embedding = self.db.encoder.encode_image(path)
                
                # 添加到列表
                image_embeddings.append(img_embedding)
                
                # 設置標籤 (圖片檔名)
                filename = os.path.basename(path)
                labels.append(filename)
            
            if not image_embeddings:
                return None
            
            # 合併所有嵌入以一起進行t-SNE
            combined_embeddings = np.vstack([
                np.array([query_embedding]),
                np.array(image_embeddings)
            ])
            
            # 應用t-SNE降維
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(combined_embeddings)-1))
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            # 分離查詢和圖片的降維結果
            query_tsne = embeddings_2d[0]
            images_tsne = embeddings_2d[1:]
            
            # 創建可視化
            plt.figure(figsize=(10, 8))
            
            # 繪製查詢嵌入
            plt.scatter(query_tsne[0], query_tsne[1], c='red', marker='*', s=200, label='Query', edgecolors='black')
            
            # 繪製圖片嵌入
            plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c='blue', label='Images', alpha=0.7)
            
            # 繪製連線 (查詢與所有圖片的連接)
            for i in range(len(images_tsne)):
                plt.plot([query_tsne[0], images_tsne[i, 0]], 
                         [query_tsne[1], images_tsne[i, 1]], 
                         'k-', alpha=0.2)
            
            # 添加圖片標籤
            for i, label in enumerate(labels):
                plt.annotate(label, 
                             xy=(images_tsne[i, 0], images_tsne[i, 1]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8)
            
            # 添加查詢標籤
            plt.annotate(f'Query: "{query}"', 
                         xy=(query_tsne[0], query_tsne[1]),
                         xytext=(10, 10),
                         textcoords='offset points',
                         fontsize=10,
                         weight='bold')
            
            plt.legend()
            plt.title('Search Query and Images t-SNE Visualization')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存圖片到內存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # 將緩衝區數據轉換為base64編碼
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            img_html = f'<img src="data:image/png;base64,{img_str}" alt="Search t-SNE Visualization">'
            
            # 關閉圖形
            plt.close()
            
            return img_html
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"生成搜尋t-SNE可視化時發生錯誤: {str(e)}\n{tb}")
            return None
    
    def get_image_details(self, evt: gr.SelectData, gallery):
        """當用戶點擊圖片時顯示詳情"""
        if not self.current_search_results or evt.index >= len(self.current_search_results):
            return "無法獲取圖片詳情"
        
        # 獲取對應的文檔
        doc = self.current_search_results[evt.index]
        doc_id = doc.metadata.get("doc_id")
        
        # 獲取圖片標籤
        tags = self.db.get_image_tags(doc_id)
        tag_names = [tag["name"] for tag in tags]
        
        # 構建詳情文本
        details = f"## 圖片詳情\n\n"
        details += f"**檔名:** {doc.metadata.get('filename', '未知')}\n\n"
        details += f"**描述:** {doc.page_content}\n\n"
        
        if tag_names:
            details += f"**標籤:** {', '.join(tag_names)}\n\n"
        
        details += f"**相似度分數:** {doc.metadata.get('score', 0):.4f}\n\n"
        details += f"**原始分數:** {doc.metadata.get('original_score', 0):.4f}\n\n"
        
        if "keyword_matches" in doc.metadata:
            details += f"**關鍵詞匹配數:** {doc.metadata.get('keyword_matches', 0)}\n\n"
        
        details += f"**路徑:** {doc.metadata.get('path', '未知')}\n\n"
        
        # 添加其他元數據
        details += "**其他信息:**\n\n"
        for key, value in doc.metadata.items():
            if key not in ["doc_id", "type", "path", "score", "original_score", "keyword_matches", "filename"]:
                details += f"- {key}: {value}\n"
        
        return details
    
    def ask_question(self, query, min_similarity=0.65):
        """基於圖片內容回答問題"""
        if not query.strip():
            return "請輸入問題"
        
        try:
            answer = self.db.qa_with_multimedia(query, min_similarity=min_similarity)
            return answer
        except Exception as e:
            return f"處理問題時發生錯誤: {str(e)}"
    
    
    
    def search_images_direct_clip(self, query, min_similarity=0.65):
        """直接使用CLIP計算文字-圖片相似度搜尋圖片，不通過Neo4j"""
        if not query.strip():
            return "請輸入搜尋關鍵詞", [], None
        
        # 檢查相似度閾值是否過高
        clip_warning = ""
        if min_similarity > 0.5:
            clip_warning = """⚠️ CLIP相似度提醒 ⚠️

使用 CLIP 模型計算文字與圖片之間的相似度時，請注意：
CLIP 是一個在極大規模、多領域資料上訓練的模型，設計目的是判斷概念上的接近程度，而不是追求非常高的絕對相似分數。

在CLIP的相似度計算中，即使是非常符合查詢概念的圖片，分數通常也只會落在 0.25 ~ 0.35之間。
這是正常現象，並不代表模型失準。

建議設定相似度閾值為 0.2~0.3 左右，而不是期待傳統語義搜尋中常見的高分（如0.8以上）。
實際使用時，請依據分數相對大小來排序與篩選結果，而不是只看分數絕對值。
"""
            min_similarity = 0.3
        
        try:
            # 1. 獲取查詢的文本嵌入
            query_embedding = self.db.encoder.encode_text(query)
            print(f"\n查詢 '{query}' 的嵌入向量:")
            print(f"維度: {query_embedding.shape}")
            print(f"嵌入向量範數: {np.linalg.norm(query_embedding)}")
            
            # 2. 獲取資料庫中的所有圖片
            all_images = self.db.get_all_images(limit=100)  # 可以調整上限
            
            # 3. 計算每張圖片的相似度
            results = []
            image_display_info = [] # 儲存用於前端顯示的路徑和分數 (path, score)
            tsne_data_points = [] # 儲存用於 T-SNE 的 (embedding, label)
            
            for image_doc in all_images:
                image_path = image_doc.metadata.get("path")
                # 從 metadata 提取圖片 embedding
                image_embedding = image_doc.metadata.get("embedding") 
                filename = image_doc.metadata.get("filename", os.path.basename(image_path) if image_path else "未知文件")
                if not image_path or not os.path.exists(image_path):
                    # 檔案可能不存在，但 embedding 可能在數據庫中
                    # 如果是直接 CLIP 搜尋，路徑不存在就無法顯示圖片，所以還是跳過
                    print(f"跳過圖片 '{filename}'：路徑 '{image_path}' 無效或不存在。")
                    continue
                
                
                # 計算余弦相似度
                # 確保兩個向量都是正規化的（通常 CLIP 的輸出是正規化的，但計算余弦相似度時明確做一次比較穩妥）
                query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
                image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
                
                similarity = np.dot(query_embedding_norm, image_embedding_norm)
                
                # 打印每個圖片的相似度分數
                # print(f"圖片: {os.path.basename(image_path)}, 相似度: {similarity:.4f}")
                
                # 如果相似度高於閾值，添加到結果
                if similarity >= min_similarity:
                    # 添加到結果文檔列表
                    image_doc.metadata["score"] = similarity # 將分數添加到文檔 metadata
                    image_doc.metadata["original_score"] = similarity
                    results.append(image_doc)

                    # 添加用於顯示的信息
                    image_display_info.append((image_path, similarity))
                    
                    # 添加用於 T-SNE 的數據點 (embedding, label)
                    tsne_data_points.append((image_embedding, filename))
            
            # 按相似度降序排序結果
            results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
            image_display_info.sort(key=lambda x: x[1], reverse=True) # 根據分數排序顯示信息
            
            # 保存結果以便顯示詳情
            self.current_search_results = results
            
            # 格式化結果訊息
            if clip_warning:
                result_text = clip_warning + "\n\n"
            else:
                result_text = ""
                
            if image_display_info:
                result_text += f"找到 {len(image_display_info)} 個相關圖片 (高於相似度閾值 {min_similarity:.4f}):"
                for i, (path, score) in enumerate(image_display_info):
                    result_text += f"\n{i+1}. {os.path.basename(path)} (相似度: {score:.4f})"
            else:
                result_text += f"沒有找到高於相似度閾值 {min_similarity:.4f} 的相關圖片。"
                # 如果沒有結果，也不需要生成 T-SNE
                return result_text, [], None
            
            # 生成查詢詞與圖片的t-SNE可視化
            tsne_html = self._generate_search_tsne_direct(query, query_embedding, image_display_info)
            
            # 返回路徑列表用於圖片展示
            return result_text, [path for path, _ in image_display_info], tsne_html
        
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"搜尋失敗: {str(e)}\n{tb}", [], None
    
    def _generate_search_tsne_direct(self, query, query_embedding, image_paths_with_scores):
        """為直接搜尋生成t-SNE可視化，使用已經計算好的嵌入"""
        try:
            if not image_paths_with_scores:
                return None
                
            image_embeddings = []
            image_scores = []
            labels = []
            
            # 為每個圖片獲取嵌入
            for path, score in image_paths_with_scores:
                if not os.path.exists(path):
                    continue
                
                # 獲取圖片嵌入
                img_embedding = self.db.encoder.encode_image(path)
                
                # 添加到列表
                image_embeddings.append(img_embedding)
                image_scores.append(score)
                
                # 設置標籤 (圖片檔名)
                filename = os.path.basename(path)
                labels.append(filename)
            
            if not image_embeddings:
                return None
            
            # 合併所有嵌入以一起進行t-SNE
            combined_embeddings = np.vstack([
                np.array([query_embedding]),
                np.array(image_embeddings)
            ])
            
            # 應用t-SNE降維
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(combined_embeddings)-1))
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            # 分離查詢和圖片的降維結果
            query_tsne = embeddings_2d[0]
            images_tsne = embeddings_2d[1:]
            
            # 創建可視化
            plt.figure(figsize=(10, 8))
            
            # 繪製查詢嵌入
            plt.scatter(query_tsne[0], query_tsne[1], c='red', marker='*', s=200, label='Query', edgecolors='black')
            
            # 創建顏色映射，基於分數的顏色漸變
            norm = plt.Normalize(min(image_scores), max(image_scores))
            cmap = plt.cm.viridis
            
            # 繪製圖片嵌入，顏色基於相似度
            sc = plt.scatter(images_tsne[:, 0], images_tsne[:, 1], 
                       c=image_scores, cmap=cmap, 
                       label='Images', alpha=0.7)
            
            # 添加顏色條
            cbar = plt.colorbar(sc)
            cbar.set_label('Similarity Score')
            
            # 繪製連線 (查詢與所有圖片的連接)
            for i in range(len(images_tsne)):
                # 線的顏色也根據相似度變化
                plt.plot([query_tsne[0], images_tsne[i, 0]], 
                         [query_tsne[1], images_tsne[i, 1]], 
                         color=cmap(norm(image_scores[i])), alpha=0.4)
            
            # 添加圖片標籤
            for i, label in enumerate(labels):
                plt.annotate(f"{label} ({image_scores[i]:.2f})", 
                             xy=(images_tsne[i, 0], images_tsne[i, 1]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8)
            
            # 添加查詢標籤
            plt.annotate(f'Query: "{query}"', 
                         xy=(query_tsne[0], query_tsne[1]),
                         xytext=(10, 10),
                         textcoords='offset points',
                         fontsize=10,
                         weight='bold')
            
            plt.legend()
            plt.title('Direct CLIP Similarity: Query and Images t-SNE Visualization')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存圖片到內存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # 將緩衝區數據轉換為base64編碼
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            img_html = f'<img src="data:image/png;base64,{img_str}" alt="Search t-SNE Visualization">'
            
            # 關閉圖形
            plt.close()
            
            return img_html
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"生成搜尋t-SNE可視化時發生錯誤: {str(e)}\n{tb}")
            return None
    
    def get_all_images_with_tags(self, selected_tags=None):
        """獲取所有圖片，可選按標籤篩選"""
        try:
            all_images = self.db.get_all_images(limit=500)
            result_docs = [] # 用來儲存原始 document 資料 (如果需要)
            gallery_items = [] # 包含 (PIL Image, title) 的列表

            if not selected_tags or len(selected_tags) == 0:
                for doc in all_images:
                    path = doc.metadata.get("path")
                    if path and os.path.exists(path):
                        try:
                            # *** 加入這行來確認路徑 ***
                            img = Image.open(path)
                            title = os.path.basename(path)
                            gallery_items.append((img.copy(), title)) # 使用 copy() 避免關閉檔案問題
                            result_docs.append(doc)
                            img.close() # 關閉檔案釋放資源
                        except Exception as e:
                            print(f"無法加載圖片 {path}: {e}")
                    elif path:
                        # *** 加入這行來確認不存在的路徑 ***
                        print(f"DEBUG: Path exists in metadata but not on disk: {path}")
                    else:
                        print("DEBUG: Path is None or empty in metadata.")
            else:
                for doc in all_images:
                    doc_id = doc.metadata.get("doc_id")
                    path = doc.metadata.get("path")
                    if not path or not os.path.exists(path):
                        continue

                    image_tags = self.db.get_image_tags(doc_id)
                    image_tag_names = [tag["name"] for tag in image_tags]

                    if all(tag in image_tag_names for tag in selected_tags):
                        try:
                            # *** 直接加載圖片為 PIL 物件 ***
                            img = Image.open(path)
                            title = os.path.basename(path)
                            gallery_items.append((img.copy(), title)) # 使用 copy()
                            result_docs.append(doc)
                            img.close() # 關閉檔案釋放資源
                        except Exception as e:
                            print(f"無法加載圖片 {path}: {e}")

            # 保存結果以便後續使用 (可能需要調整儲存的內容)
            self.current_gallery_images = result_docs # 或者儲存其他你需要的信息

            result_text = f"顯示 {len(gallery_items)} 張圖片"

            # *** 返回 PIL 物件列表給 Gallery ***
            # Gallery 會自動處理 (PIL Image, caption) 的元組
            return result_text, gallery_items, [item[1] for item in gallery_items] # 第三個返回值是圖片標題列表

        except Exception as e:
            tb = traceback.format_exc()
            return f"獲取圖片時發生錯誤: {str(e)}\n{tb}", [], []
    
    def get_gallery_image_details(self, evt: gr.SelectData, gallery):
        """當用戶在標籤畫廊中點擊圖片時顯示詳情"""
        if not self.current_gallery_images or evt.index >= len(self.current_gallery_images):
            return "無法獲取圖片詳情", [], []
        
        # 獲取對應的文檔
        doc = self.current_gallery_images[evt.index]
        doc_id = doc.metadata.get("doc_id")
        
        # 獲取圖片標籤
        tags = self.db.get_image_tags(doc_id)
        tag_names = [tag["name"] for tag in tags]
        
        # 構建詳情文本
        details = f"## 圖片詳情\n\n"
        details += f"**檔名:** {doc.metadata.get('filename', '未知')}\n\n"
        details += f"**ID:** {doc_id}\n\n"
        details += f"**描述:** {doc.page_content}\n\n"
        
        if tag_names:
            details += f"**標籤:** {', '.join(tag_names)}\n\n"
        
        details += f"**路徑:** {doc.metadata.get('path', '未知')}\n\n"
        
        # 添加其他元數據
        details += "**其他信息:**\n\n"
        for key, value in doc.metadata.items():
            if key not in ["doc_id", "type", "path", "filename", "embedding"]:
                details += f"- {key}: {value}\n"
        
        # 查找相似圖片
        try:
            similar_gallery_images = []  # 包含圖片和標題的列表
            
            # 使用Neo4j查詢與當前圖片相似的圖片
            with self.db.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (img:MultimediaContent:Image {id: $image_id})-[r:SIMILAR]->(similar:MultimediaContent:Image)
                    RETURN similar.id as id, similar.path as path, r.score as similarity
                    ORDER BY r.score DESC
                    LIMIT 10
                """, {"image_id": doc_id})
                
                for record in result:
                    path = record["path"]
                    if path and os.path.exists(path):
                        # 加載為PIL圖片對象
                        try:
                            cache_path = self.copy_to_gradio_tmp(path)
                            print("Gallery paths:", cache_path)
                            title = f"{os.path.basename(path)} (相似度: {record['similarity']:.4f})"
                            similar_gallery_images.append((cache_path, title))
                        except Exception as e:
                            print(f"無法加載相似圖片 {path}: {e}")
            
            # 返回結果
            return details, similar_gallery_images, [item[1] for item in similar_gallery_images]
        except Exception as e:
            return f"{details}\n\n獲取相似圖片時發生錯誤: {str(e)}", [], []
    
    def delete_image(self, image_id):
        """從檔案系統和Neo4j中刪除圖片"""
        if not image_id:
            return "請提供要刪除的圖片ID"
        
        try:
            # 首先獲取圖片的路徑
            path = None
            with self.db.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (img:MultimediaContent:Image {id: $image_id})
                    RETURN img.path as path
                """, {"image_id": image_id})
                record = result.single()
                if record:
                    path = record["path"]
            
            if not path:
                return f"找不到ID為 {image_id} 的圖片"
                
            # 從Neo4j中刪除圖片（使用DETACH DELETE）
            with self.db.driver.session(database=NEO4J_DATABASE) as session:
                session.run("""
                    MATCH (img:MultimediaContent:Image {id: $image_id})
                    DETACH DELETE img
                """, {"image_id": image_id})
            
            # 從檔案系統中刪除圖片
            if os.path.exists(path):
                os.remove(path)
                file_deleted = True
            else:
                file_deleted = False
            
            # 返回結果
            if file_deleted:
                return f"圖片 {image_id} 已成功從資料庫和檔案系統中刪除"
            else:
                return f"圖片 {image_id} 已從資料庫中刪除，但檔案 {path} 不存在或無法刪除"
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"刪除圖片時發生錯誤: {str(e)}\n{tb}"
        
    def copy_to_gradio_tmp(self, src_path):
        """將圖片複製到 Gradio 快取資料夾，回傳新路徑"""
        if not os.path.exists(src_path):
            return None
        # 直接用 GRADIO_TEMP_DIR
        cache_dir = os.environ.get('GRADIO_TEMP_DIR', os.path.join(tempfile.gettempdir(), "gradio"))
        os.makedirs(cache_dir, exist_ok=True)
        ext = os.path.splitext(src_path)[1]
        dst_path = os.path.join(cache_dir, f"{uuid.uuid4().hex}{ext}")
        shutil.copy(src_path, dst_path)
        return dst_path
    
    def build_ui(self):
        """構建 Gradio 介面"""
        with gr.Blocks(title="多媒體圖像搜尋系統", theme=gr.themes.Soft(primary_hue="blue")) as demo:
            gr.Markdown("# 多媒體圖像搜尋系統")
            
            with gr.Tabs():
                # 上傳頁面
                with gr.TabItem("上傳圖片"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            upload_image = gr.Image(label="選擇圖片上傳", type="pil", image_mode="RGB")
                            description = gr.Textbox(label="圖片描述（可選）", placeholder="請輸入圖片描述...")
                            
                            with gr.Row():
                                use_auto_tagging = gr.Checkbox(label="使用AI自動標記", value=True)
                            
                            with gr.Accordion("標籤選項", open=True):
                                tag_choices = gr.Dropdown(
                                    label="選擇現有標籤", 
                                    choices=[],
                                    multiselect=True,
                                    allow_custom_value=True
                                )
                                new_tags = gr.Textbox(
                                    label="添加新標籤（用逗號、分號或空格分隔）", 
                                    placeholder="例如: 風景, 山, 海..."
                                )
                                
                                def update_tag_choices():
                                    return gr.Dropdown(choices=self.get_all_tags_for_ui())
                                
                                refresh_tags_btn = gr.Button("刷新標籤列表", variant="secondary", size="sm")
                                refresh_tags_btn.click(fn=update_tag_choices, outputs=tag_choices)
                            
                            upload_btn = gr.Button("上傳圖片", variant="primary")
                        
                        with gr.Column(scale=1):
                            upload_result = gr.Textbox(label="上傳結果", elem_id="upload_result")
                            preview_image = gr.Image(label="預覽", type="filepath")
                            current_tags = gr.Dataframe(
                                headers=["標籤"],
                                datatype=["str"],
                                label="已添加標籤"
                            )
                            
                    # 顯示詳細的錯誤信息（如果有）
                    error_box = gr.Markdown(visible=False, elem_id="error_box")
                    
                    def handle_upload(image, description, tag_choices, new_tags, use_auto_tagging):
                        selected_tags = tag_choices if isinstance(tag_choices, list) else []
                        result, path, tag_names = self.upload_image(
                            image, description, selected_tags, new_tags, use_auto_tagging
                        )
                        
                        # 準備標籤表格
                        tags_df = [[tag] for tag in tag_names] if tag_names else []
                        
                        if "失敗" in result:
                            return result, path, tags_df, result, "visible", update_tag_choices()
                        return result, path, tags_df, "", "hidden", update_tag_choices()
                    
                    upload_btn.click(
                        fn=handle_upload,
                        inputs=[upload_image, description, tag_choices, new_tags, use_auto_tagging],
                        outputs=[upload_result, preview_image, current_tags, error_box, error_box, tag_choices]
                    )
                    
                    # 頁面載入時更新標籤列表
                    demo.load(fn=self.get_all_tags_for_ui, outputs=tag_choices)
                
                # 搜尋頁面
                with gr.TabItem("搜尋圖片"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_query = gr.Textbox(label="搜尋關鍵詞", placeholder="輸入搜尋詞...")
                            similarity_threshold = gr.Slider(
                                minimum=0.1, 
                                maximum=0.9, 
                                value=0.65, 
                                step=0.05, 
                                label="最低相似度閾值"
                            )
                            with gr.Row():
                                search_btn = gr.Button("Neo4j搜尋", variant="primary")
                                direct_search_btn = gr.Button("CLIP搜尋", variant="secondary")
                            search_result = gr.Textbox(label="搜尋結果")
                        
                        with gr.Column(scale=2):
                            gallery = gr.Gallery(
                                label="搜尋結果",
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                            details = gr.Markdown(label="圖片詳情")
                    
                    with gr.Row():
                        search_tsne = gr.HTML(label="查詢-圖片關係可視化", visible=True)
                    
                    search_btn.click(
                        fn=self.search_images,
                        inputs=[search_query, similarity_threshold],
                        outputs=[search_result, gallery, search_tsne]
                    )
                    
                    direct_search_btn.click(
                        fn=self.search_images_direct_clip,
                        inputs=[search_query, similarity_threshold],
                        outputs=[search_result, gallery, search_tsne]
                    )
                    
                    gallery.select(
                        fn=self.get_image_details,
                        inputs=[gallery],
                        outputs=[details]
                    )
                
                # 新增：標籤畫廊頁面
                with gr.TabItem("標籤畫廊"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gallery_tag_choices = gr.Dropdown(
                                label="按標籤篩選（選擇多個標籤為AND關係）", 
                                choices=self.get_all_tags_for_ui(),
                                multiselect=True
                            )
                            
                            refresh_gallery_btn = gr.Button("刷新畫廊", variant="primary")
                            gallery_result = gr.Textbox(label="結果")
                            
                            with gr.Accordion("刪除圖片", open=False):
                                delete_image_id = gr.Textbox(label="輸入要刪除的圖片ID", placeholder="img_...")
                                delete_btn = gr.Button("刪除圖片", variant="stop")
                                delete_result = gr.Textbox(label="刪除結果")
                                
                                delete_btn.click(
                                    fn=self.delete_image,
                                    inputs=[delete_image_id],
                                    outputs=[delete_result]
                                )
                        
                        with gr.Column(scale=2):
                            all_images_gallery = gr.Gallery(
                                label="所有圖片",
                                columns=4,
                                object_fit="contain",
                                height="auto"
                            )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gallery_details = gr.Markdown(label="圖片詳情")
                        
                        with gr.Column(scale=1):
                            similar_images_gallery = gr.Gallery(
                                label="相似圖片",
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                    
                    # 刷新畫廊按鈕事件
                    refresh_gallery_btn.click(
                        fn=self.get_all_images_with_tags,
                        inputs=[gallery_tag_choices],
                        outputs=[gallery_result, all_images_gallery]
                    )
                    
                    # 圖片點擊事件
                    all_images_gallery.select(
                        fn=self.get_gallery_image_details,
                        inputs=[all_images_gallery],
                        outputs=[gallery_details, similar_images_gallery]
                    )
                    
                    # 頁面載入時更新畫廊
                    demo.load(
                        fn=lambda: self.get_all_images_with_tags(),
                        inputs=None,
                        outputs=[gallery_result, all_images_gallery]
                    )
                
                # 圖片問答頁面
                with gr.TabItem("圖片問答"):
                    with gr.Row():
                        with gr.Column():
                            qa_query = gr.Textbox(label="問題", placeholder="請輸入與圖片相關的問題...")
                            qa_similarity = gr.Slider(
                                minimum=0.1, 
                                maximum=0.9, 
                                value=0.65, 
                                step=0.05, 
                                label="最低相似度閾值"
                            )
                            qa_btn = gr.Button("提問", variant="primary")
                            qa_result = gr.Markdown(label="回答")
                    
                    qa_btn.click(
                        fn=self.ask_question,
                        inputs=[qa_query, qa_similarity],
                        outputs=[qa_result]
                    )
            
            gr.Markdown("### 注意事項\n"
                      "1. 上傳圖片後會自動添加到資料庫\n"
                      "2. 搜尋時可調整相似度閾值以獲得更精確的結果\n"
                      "3. 點擊圖片可查看詳細資訊\n"
                      "4. 問答頁面可基於圖片內容回答問題\n"
                      "5. 嵌入空間可視化使用t-SNE演算法將高維嵌入降到2D空間，藍色點表示圖片嵌入，紅色點表示文本嵌入，"
                      "連線表示相對應的圖片-文本對。緊密聚集和短連線表示圖片和文本嵌入較好地對齊。\n"
                      "6. 使用直接CLIP搜尋時，請將相似度閾值設定較低（建議0.2 ~ 0.3），因為CLIP模型的相似度分數通常較低，"
                      "即使是非常符合查詢概念的圖片，分數通常也只會落在0.25 ~ 0.35之間。這是正常現象，實際使用時應依據分數的相對大小來排序結果。")
        
        return demo
    
    def launch(self, share=False, debug=False, server_port=None):
        """啟動 Gradio 介面"""
        demo = self.build_ui()
        demo.launch(share=share, debug=debug, server_port=server_port)
    
    def __del__(self):
        """清理臨時檔案"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"清理臨時檔案時發生錯誤: {e}")

    def get_all_tags_for_ui(self):
        """獲取所有標籤供UI選擇"""
        tags = self.db.get_all_tags()
        return [tag["name"] for tag in tags]


if __name__ == "__main__":
    app = ImageDatabaseUI()
    app.launch(debug=True) 