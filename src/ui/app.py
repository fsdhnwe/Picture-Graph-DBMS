import os
import sys
import gradio as gr
import tempfile
from pathlib import Path
import shutil
import numpy as np
import time
from PIL import Image

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 獲取 src 目錄
project_root = os.path.dirname(current_dir)  # 獲取項目根目錄
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用絕對路徑導入
from src.database.neo4j_graph_rag import Neo4jGraphRAG

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
    
    def upload_image(self, image, description=None):
        """上傳圖片到資料庫"""
        if image is None:
            return "請選擇圖片上傳", None
        
        # 保存圖片到測試圖片目錄
        timestamp = int(time.time())
        
        try:
            if isinstance(image, str):  # 已經是路徑了
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
                    return f"不支援的圖片格式: {type(image)}", None
            
            # 準備元數據
            metadata = {}
            if description:
                metadata["user_description"] = description
            
            # 添加到資料庫
            doc_id = self.db.add_image(save_path, metadata)
            return f"圖片上傳成功！ID: {doc_id}", save_path
        except Exception as e:
            return f"圖片上傳失敗: {str(e)}", None
    
    def search_images(self, query, min_similarity=0.65):
        """搜尋相似圖片"""
        if not query.strip():
            return "請輸入搜尋關鍵詞", []
        
        try:
            # 執行搜尋
            results = self.db.search(query, k=9, min_similarity=min_similarity)
            
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
            
            # 只返回路徑列表用於圖片展示
            return result_text, [path for path, _ in image_paths]
        
        except Exception as e:
            return f"搜尋失敗: {str(e)}", []
    
    def get_image_details(self, evt: gr.SelectData, gallery):
        """當用戶點擊圖片時顯示詳情"""
        if not self.current_search_results or evt.index >= len(self.current_search_results):
            return "無法獲取圖片詳情"
        
        # 獲取對應的文檔
        doc = self.current_search_results[evt.index]
        
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
                            upload_btn = gr.Button("上傳圖片", variant="primary")
                        
                        with gr.Column(scale=1):
                            upload_result = gr.Textbox(label="上傳結果", elem_id="upload_result")
                            preview_image = gr.Image(label="預覽", type="filepath")
                            
                    # 顯示詳細的錯誤信息（如果有）
                    error_box = gr.Markdown(visible=False, elem_id="error_box")
                    
                    def handle_upload(image, description):
                        result, path = self.upload_image(image, description)
                        if "失敗" in result:
                            return result, path, result, "visible"
                        return result, path, "", "hidden"
                    
                    upload_btn.click(
                        fn=handle_upload,
                        inputs=[upload_image, description],
                        outputs=[upload_result, preview_image, error_box, error_box]
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
                            search_btn = gr.Button("搜尋", variant="primary")
                            search_result = gr.Textbox(label="搜尋結果")
                        
                        with gr.Column(scale=2):
                            gallery = gr.Gallery(
                                label="搜尋結果",
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                            details = gr.Markdown(label="圖片詳情")
                    
                    search_btn.click(
                        fn=self.search_images,
                        inputs=[search_query, similarity_threshold],
                        outputs=[search_result, gallery]
                    )
                    
                    gallery.select(
                        fn=self.get_image_details,
                        inputs=[gallery],
                        outputs=[details]
                    )
                
                # 問答頁面
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
                      "4. 問答頁面可基於圖片內容回答問題")
        
        return demo
    
    def launch(self, share=False, debug=False, server_port=7860):
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


if __name__ == "__main__":
    app = ImageDatabaseUI()
    app.launch(debug=True) 