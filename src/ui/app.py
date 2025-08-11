import os
import gradio as gr
import tempfile
import shutil
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import io
from sklearn.manifold import TSNE
import base64
import tempfile
import traceback
import datetime


# 使用絕對路徑導入
from src.database.clip_milvus import ClipMilvus

class ImageDatabaseUI:
    def __init__(self):
        # 初始化資料庫連接
        milvus_config = {
            'host': 'localhost',
            'port': '19530'
        }
        self.db = ClipMilvus(use_local_llm=True, milvus_config=milvus_config)
        
        # 創建臨時目錄存放上傳圖片
        self.temp_dir = tempfile.mkdtemp()
        
        # 圖片儲存路徑
        self.image_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(self.image_dir, exist_ok=True)
        # 影片儲存路徑
        self.video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        # 搜尋結果暫存
        self.current_search_results = []
        
        # 畫廊結果暫存
        self.current_gallery_images = []
    
    def upload_files(self, files, description=None):
        """上傳多張圖片或影片到資料庫
        
        Args:
            files: Gradio File objects (list)
            description: 檔案描述（可選）
            
        Returns:
            tuple: (結果訊息, 檔案路徑列表)
        """
        if not files:
            return "請選擇檔案上傳", []
            
        results = []
        saved_media_paths = [] # To store paths for gallery preview
        timestamp = int(time.time())
        
        for i, file_obj in enumerate(files):
            try:
                original_filename = os.path.basename(file_obj.name) # Gradio provides temp path in file_obj.name
                extension = os.path.splitext(original_filename)[1].lower()
                
                # 根據副檔名決定存放資料夾
                if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    save_dir = self.image_dir
                elif extension in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                    save_dir = self.video_dir
                else:
                    save_dir = self.image_dir  # 預設仍存圖片資料夾

                save_path = os.path.join(save_dir, f"media_{timestamp}_{i}{extension}")
                
                # Copy the uploaded file from its temporary location to the save_path
                shutil.copy(file_obj.name, save_path)

                metadata = {}
                if description:
                    metadata["user_description"] = description
                
                doc_id = None
                if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    doc_id = self.db.add_image(save_path, metadata, None)
                    results.append(f"圖片 {original_filename} 上傳成功！ID: {doc_id}")
                elif extension in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                    # Assuming you have an add_video method similar to add_image
                    doc_id = self.db.add_video(save_path, metadata) 
                    results.append(f"影片 {original_filename} 上傳成功！ID: {doc_id}")
                else:
                    results.append(f"檔案 {original_filename}: 不支援的格式 {extension}")
                    os.remove(save_path) # Remove unsupported file
                    continue
                
                if doc_id:
                    saved_media_paths.append(save_path)
                
            except Exception as e:
                results.append(f"檔案 {original_filename} 上傳失敗: {str(e)}")
        
        if not saved_media_paths:
            return "所有檔案上傳失敗！\\n" + "\\n".join(results), []
        elif len(saved_media_paths) < len(files):
            return f"部分檔案上傳成功 ({len(saved_media_paths)}/{len(files)})。\\n" + "\\n".join(results), saved_media_paths
        else:
            return f"所有檔案上傳成功！\\n" + "\\n".join(results), saved_media_paths
    
    
    def get_image_details(self, evt: gr.SelectData, gallery):
        """當用戶點擊圖片時顯示詳情"""
        if not self.current_search_results or evt.index >= len(self.current_search_results):
            return "無法獲取圖片詳情"
        
        # 獲取對應的文檔
        doc = self.current_search_results[evt.index]
        doc_id = doc.metadata.get("doc_id")
        
        # 構建詳情文本
        details = f"## 圖片詳情\n\n"
        details += f"**檔名:** {doc.metadata.get('filename', '未知')}\n\n"
        details += f"**描述:** {doc.page_content}\n\n"
        
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
        """直接使用CLIP計算文字-圖片相似度搜尋圖片"""
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
                try:
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
                except Exception as e:
                    print(f"計算圖片 '{filename}' 的相似度時發生錯誤: {e}")
            
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
            
            # 返回路徑列表用於圖片展示
            return result_text, [path for path, _ in image_display_info]
        
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"搜尋失敗: {str(e)}\n{tb}", [], None
    
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"生成搜尋t-SNE可視化時發生錯誤: {str(e)}\n{tb}")
            return None
    
    def get_all_media(self, media_type="圖片"): 
        """獲取所有圖片或影片"""
        try:
            items = []
            if media_type == "圖片":
                items = self.db.get_all_images(limit=500)
            elif media_type == "影片":
                items = self.db.get_all_videos(limit=500)

            result_docs = [] 
            gallery_items = [] 

            for doc in items:
                path = doc.metadata.get("path")
                if path and os.path.exists(path):
                    try:
                        title = doc.page_content or doc.metadata.get('description') or os.path.basename(path)
                        if media_type == "圖片":
                            img = Image.open(path)
                            gallery_items.append((path, title)) 
                            img.close() 
                        elif media_type == "影片":
                            # For videos, Gallery can take file paths directly
                            gallery_items.append((path, title)) 
                        result_docs.append(doc)
                    except Exception as e:
                        print(f"無法加載媒體 {path}: {e}")
                elif path:
                    print(f"DEBUG: Path exists in metadata but not on disk: {path}")
                else:
                    print("DEBUG: Path is None or empty in metadata.")
                            

            # 保存結果以便後續使用 (可能需要調整儲存的內容)
            self.current_gallery_images = result_docs # 或者儲存其他你需要的信息

            result_text = f"顯示 {len(gallery_items)} 個{media_type}"

            # For images, gallery_items are (PIL.Image, title)
            # For videos, gallery_items are (filepath, title)
            # Gradio Gallery handles both.
            return result_text, gallery_items

        except Exception as e:
            tb = traceback.format_exc()
            return f"獲取{media_type}時發生錯誤: {str(e)}\\n{tb}", []

    def get_gallery_item_details(self, evt: gr.SelectData, gallery):
        """當用戶在標籤畫廊中點擊媒體時顯示詳情"""
        if not self.current_gallery_images or evt.index >= len(self.current_gallery_images):
            return "無法獲取媒體詳情", [], []
        
        doc = self.current_gallery_images[evt.index]
        doc_id = doc.metadata.get("doc_id")
        media_type = doc.metadata.get("type", "未知類型") # image or video

        details = f"## {media_type.capitalize()} 詳情\n\n"
        details += f"**檔名/標題:** {doc.metadata.get('filename') or doc.metadata.get('title', '未知')}\n\n"
        details += f"**ID:** {doc_id}\n\n"
        
        description = doc.page_content or doc.metadata.get('description', '')
        details += f"**描述:** {description}\n\n"
        details += f"**路徑:** {doc.metadata.get('path', '未知')}\n\n"
        
        details += "**其他信息:**\n\n"
        for key, value in doc.metadata.items():
            if key not in ["doc_id", "type", "path", "filename", "title", "description", "embedding"]:
                if key == "created_time" or key == "updated_time":
                    # 格式化時間戳
                    value = datetime.datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                details += f"- {key}: {value}\n"
        
        # 查找相似圖片
        try:
            similar_gallery_images = []  # 包含圖片和標題的列表
            
            # 使用Milvus查詢與當前圖片相似的圖片
            # 1. 首先獲取當前圖片的cluster_id
            cluster_id = doc.metadata.get("cluster_id")
            
            if cluster_id:
                # 2. 獲取相同cluster_id的圖片
                similar_docs = self.db.get_images_by_cluster(cluster_id, limit=10)
                
                # 過濾掉當前圖片自身
                similar_docs = [d for d in similar_docs if d.metadata.get("doc_id") != doc_id]
                
                # 如果找到相似圖片
                if similar_docs:
                    for similar_doc in similar_docs:
                        path = similar_doc.metadata.get("path")
                        if path and os.path.exists(path):
                            try:
                                filename = similar_doc.metadata.get("filename", os.path.basename(path))
                                title = f"{filename} (同群組: {cluster_id})"
                                similar_gallery_images.append((path, title))
                            except Exception as e:
                                print(f"無法加載相似圖片 {path}: {e}")
            
            # 返回結果
            return details, similar_gallery_images, [item[1] for item in similar_gallery_images]
        except Exception as e:
            return f"{details}\n\n獲取相似圖片時發生錯誤: {str(e)}", [], []
    
    def delete_image(self, image_id):
        """從檔案系統和Milvus中刪除圖片"""
        if not image_id:
            return "請提供要刪除的圖片ID"
        
        try:
            # 從所有圖片中查找匹配的ID
            all_images = self.db.get_all_images(limit=1000)
            target_image = None
            
            for img in all_images:
                if img.metadata.get("doc_id") == image_id:
                    target_image = img
                    break
            
            if not target_image:
                return f"找不到ID為 {image_id} 的圖片"
            
            path = target_image.metadata.get("path")
            
            # 從Milvus中刪除圖片
            db_deleted = self.db.delete_image(image_id)
            
            # 從檔案系統中刪除圖片
            if path and os.path.exists(path):
                os.remove(path)
                file_deleted = True
            else:
                file_deleted = False
            
            # 返回結果
            if db_deleted and file_deleted:
                return f"圖片 {image_id} 已成功從資料庫和檔案系統中刪除"
            elif db_deleted:
                return f"圖片 {image_id} 已從資料庫中刪除，但檔案不存在或無法刪除"
            elif file_deleted:
                return f"圖片 {image_id} 檔案已從檔案系統中刪除，但資料庫刪除失敗"
            else:
                return f"圖片 {image_id} 刪除失敗：資料庫和檔案系統都未能刪除"
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"刪除圖片時發生錯誤: {str(e)}\n{tb}"
        
    def build_ui(self):
        """構建 Gradio 介面"""
        with gr.Blocks(title="多媒體圖像搜尋系統", theme=gr.themes.Soft(primary_hue="blue")) as demo:
            gr.Markdown("# 多媒體圖像搜尋系統")
            
            with gr.Tabs():
                # 上傳頁面
                with gr.TabItem("上傳圖片/影片"): # Changed tab name
                    with gr.Row():
                        with gr.Column(scale=2):
                            upload_files_component = gr.Files(label="選擇圖片或影片上傳", file_count="multiple", file_types=["image", "video"])
                            description_input = gr.Textbox(label="檔案描述（可選）", placeholder="請輸入檔案描述...")
                            
                            upload_btn = gr.Button("上傳檔案", variant="primary")
                        
                        with gr.Column(scale=1):
                            upload_result_text = gr.Textbox(label="上傳結果", elem_id="upload_result")
                            preview_gallery_component = gr.Gallery(label="預覽", columns=2, height=300)
                            

                    # 顯示詳細的錯誤信息（如果有）
                    error_box = gr.Markdown(visible=False, elem_id="error_box")
                    
                    def handle_upload_files(files, description): # Renamed function and params
                        result, paths = self.upload_files(files, description) # Call new method
                        
                        # Preview gallery expects list of (data, caption) or just data.
                        # For local files (images/videos), paths should work.
                        preview_items = []
                        if paths:
                            for p in paths:
                                preview_items.append((p, os.path.basename(p)))

                        if "失敗" in result or not paths:
                            # If paths is empty but result indicates partial success, it's an issue.
                            # For simplicity, if any failure or no paths, show error.
                            return result, [], result, gr.Markdown(visible=True)
                        return result, preview_items, "", gr.Markdown(visible=False)
                    
                    upload_btn.click(
                        fn=handle_upload_files,
                        inputs=[upload_files_component, description_input],
                        outputs=[upload_result_text, preview_gallery_component, error_box, error_box] # error_box output needs to be a component
                    )
                    
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
                                direct_search_btn = gr.Button("CLIP搜尋")
                            search_result = gr.Textbox(label="搜尋結果")
                        
                        with gr.Column(scale=2):
                            gallery = gr.Gallery(
                                label="搜尋結果",
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                            details = gr.Markdown(label="圖片詳情")
                    
                    
                    direct_search_btn.click(
                        fn=self.search_images_direct_clip,
                        inputs=[search_query, similarity_threshold],
                        outputs=[search_result, gallery]
                    )
                    
                    gallery.select(
                        fn=self.get_image_details,
                        inputs=[gallery],
                        outputs=[details]
                    )
                
                # 新增：標籤畫廊頁面
                with gr.TabItem("媒體畫廊"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            media_type_choice = gr.Radio(choices=["圖片", "影片"], label="選擇媒體類型", value="圖片")
                            refresh_gallery_btn = gr.Button("刷新畫廊", variant="primary")
                            gallery_result_text = gr.Textbox(label="結果")
                            
                            with gr.Accordion("刪除媒體", open=False):
                                delete_media_id_input = gr.Textbox(label="輸入要刪除的媒體ID", placeholder="img_... or vid_...") # Renamed
                                delete_btn = gr.Button("刪除媒體", variant="stop")
                                delete_result_text = gr.Textbox(label="刪除結果")
                                
                                delete_btn.click(
                                    fn=self.delete_image,
                                    inputs=[delete_media_id_input],
                                    outputs=[delete_result_text]
                                )
                        
                        with gr.Column(scale=2):
                            all_media_gallery = gr.Gallery(
                                label="所有媒體",
                                columns=4,
                                object_fit="contain",
                                height="auto"
                            )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gallery_item_details_md = gr.Markdown(label="媒體詳情")
                        
                        with gr.Column(scale=1):
                            similar_items_gallery = gr.Gallery( 
                                label="相似媒體",
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                    
                    refresh_gallery_btn.click(
                        fn=self.get_all_media, # Call new method
                        inputs=[media_type_choice],  
                        outputs=[gallery_result_text, all_media_gallery]
                    )
                    
                    all_media_gallery.select(
                        fn=self.get_gallery_item_details, # Call new method
                        inputs=[all_media_gallery], # This input is just the gallery component, evt.index is used inside
                        outputs=[gallery_item_details_md, similar_items_gallery] # Pass to new similar_items_gallery
                    )
                    
                    def initial_load_gallery(media_type):
                        return self.get_all_media(media_type=media_type)

                    demo.load(
                        fn=initial_load_gallery,
                        inputs=[media_type_choice], # Use the current value of the radio button
                        outputs=[gallery_result_text, all_media_gallery]
                    )
                
                # 圖片問答頁面
                with gr.TabItem("圖片問答"):
                    with gr.Row():
                        with gr.Column():
                            qa_query = gr.Textbox(label="問題", placeholder="請輸入與圖片相關的問題...")
                            qa_similarity = gr.Slider(
                                minimum=0.1, 
                                maximum=0.9, 
                                value=0.25, 
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