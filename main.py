from src.database.clip_milvus import MilvusGraphRAG
import os
import time

def add_images_to_db(graph_rag, image_dir):
    """將圖片添加到數據庫"""
    print(f"處理圖片目錄: {image_dir}")
    
    os.makedirs(image_dir, exist_ok=True)
    image_ids = []
    
    # 添加圖片
    print("添加圖片到圖數據庫...")
    # 只處理指定的兩張圖片
    target_images = ["phone.jpg"]
    
    for image_file in target_images:
        image_path = os.path.join(image_dir, image_file)
        if os.path.exists(image_path):
            print(f"處理圖片: {image_path}")
            img_id = graph_rag.add_image(image_path, {"tags": "test image", "filename": image_file})
            image_ids.append(img_id)
    
    # 如果有多於一張圖片，創建一些關係
    if len(image_ids) >= 2:
        for i in range(len(image_ids) - 1):
            graph_rag.create_relationship(
                image_ids[i], image_ids[i+1], "NEXT_IMAGE", 
                {"created_at": time.time()}
            )
            print(f"建立關係: {image_ids[i]} -> {image_ids[i+1]}")
    
    return image_ids

def search_mode(graph_rag):
    """互動式搜索模式"""
    print("\n進入互動式搜索模式。輸入 'q' 或 'exit' 退出。")
    
    while True:
        query = input("\n請輸入搜索詞 (q 退出): ")
        if query.lower() in ('q', 'exit', 'quit'):
            break
            
        print(f"搜索: '{query}'")
        results = graph_rag.search(query)
        
        if not results:
            print("未找到相關圖片")
            continue
            
        print(f"找到 {len(results)} 個相關項目:")
        
        for i, doc in enumerate(results):
            doc_type = doc.metadata.get("type", "unknown")
            score = doc.metadata.get("score", 0)
            
            if doc_type == "image":
                path = doc.metadata.get("path", "unknown")
                print(f"{i+1}. 相似度: {score:.4f} - 圖片路徑: {path}")
            else:
                title = doc.metadata.get("title", "untitled")
                print(f"{i+1}. 相似度: {score:.4f} - {doc_type.capitalize()}: {title}")

def test_predefined_queries(graph_rag):
    """測試預定義的查詢"""
    search_queries = [
        "stairs in nature",
        "stone steps"
    ]
    
    for query in search_queries:
        print(f"\n執行文本搜索: '{query}'")
        results = graph_rag.search(query)
        print(f"找到 {len(results)} 個相關項目:")
        
        for i, doc in enumerate(results):
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type == "image":
                path = doc.metadata.get("path", "unknown")
                print(f"{i+1}. 圖片: {os.path.basename(path)}, 相似度得分: {doc.metadata.get('score', 'N/A'):.4f}")
            else:
                print(f"{i+1}. {doc_type.capitalize()}: {doc.metadata.get('title', 'untitled')}")

def setup_neo4j_connection():
    """設置連接模式"""
    print("\n選擇運行模式:")
    print("1. 模擬模式 (無需Neo4j連接)")
    print("2. 真實Neo4j模式 (需要連接到Neo4j資料庫)")
    
    choice = input("選擇模式 (1/2): ").strip()
    
    use_mock = True if choice != "2" else False
    
    if not use_mock:
        print("\n正在嘗試連接Neo4j資料庫...")
        print("確保Neo4j伺服器正在運行，且設置正確")
        print("當前連接設置:")
        from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        print(f"URI: {NEO4J_URI}")
        print(f"用戶名: {NEO4J_USERNAME}")
        print(f"密碼: {'*' * len(NEO4J_PASSWORD)}")
        
        confirm = input("\n設置正確? (y/n): ").strip().lower()
        if confirm != 'y':
            print("請修改 src/config.py 文件中的設置後重新運行")
            exit(0)
    
    return use_mock

def list_all_images(graph_rag):
    """列出數據庫中的所有圖片"""
    if graph_rag.use_mock:
        print("\n模擬模式中的圖片:")
        for i, doc in enumerate(graph_rag.mock_docs):
            if doc.get("type") == "image":
                print(f"{i+1}. ID: {doc['id']}, 檔名: {doc['filename']}, 路徑: {doc['path']}")
        return
    
    # 從Neo4j獲取所有圖片
    try:
        from src.config import NEO4J_DATABASE
        with graph_rag.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (n:MultimediaContent:Image)
                RETURN n.id as id, n.filename as filename, n.path as path
                ORDER BY n.filename
            """)
            
            print("\nNeo4j數據庫中的圖片:")
            i = 0
            for record in result:
                i += 1
                print(f"{i}. ID: {record['id']}, 檔名: {record['filename']}, 路徑: {record['path']}")
            
            if i == 0:
                print("數據庫中沒有圖片")
    except Exception as e:
        print(f"獲取圖片列表時發生錯誤: {e}")

def main():
    # 設置連接模式
    use_mock = setup_neo4j_connection()
    
    # 初始化 - 根據選擇使用模擬模式或真實模式
    graph_rag = MilvusGraphRAG(use_local_llm=False, use_mock=use_mock)

    # 測試目錄
    image_dir = "./test_images"
    
    # 顯示選項
    print("\n選擇操作:")
    print("1. 添加圖片到數據庫")
    print("2. 查看所有圖片")
    print("3. 搜索圖片")
    
    choice = input("選擇 (1/2/3): ").strip()
    
    if choice == "1":
        # 添加圖片
        add_images_to_db(graph_rag, image_dir)
    elif choice == "2":
        # 列出所有圖片
        list_all_images(graph_rag)
    else:
        # 搜索功能
        print("\n選擇搜索模式:")
        print("1. 測試預定義查詢")
        print("2. 互動搜索模式")
        
        search_choice = input("選擇 (1/2): ").strip()
        
        if search_choice == "1":
            test_predefined_queries(graph_rag)
        else:
            search_mode(graph_rag)

if __name__ == "__main__":
    print("啟動跨模態圖像-文本搜索系統...")
    main()
    print("程序執行完畢。")