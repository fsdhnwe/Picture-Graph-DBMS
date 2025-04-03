from src.database.neo4j_graph_rag import Neo4jGraphRAG


        

    
# 实例使用示例
def example_usage():
    # 初始化 - 使用本地模型
    graph_rag = Neo4jGraphRAG(use_local_llm=True)
    
    # 添加图片
    img_id1 = graph_rag.add_image("pngtree-stairs-famous-road-stairs-photo-image_1094418.jpg", {"tags": "landscape, nature, stairs", "location": "none"})
    img_id2 = graph_rag.add_image("stairs-7245353_1280.jpg", {"tags": "landscape, flowerpot, stairs", "location": "none"})
    
    # 添加视频
    # video_id = graph_rag.add_video("path/to/video.mp4", {"title": "Conference Presentation", "speaker": "Dr. Smith"})
    
    # 添加YouTube视频
    # yt_id = graph_rag.add_youtube_video("https://www.youtube.com/watch?v=example", {"category": "Educational"})
    
    # 创建关系
    graph_rag.create_relationship(img_id1, img_id2, "SIMILAR_TO", {"similarity_score": 0.85})
    # graph_rag.create_relationship(video_id, yt_id, "RELATED_CONTENT", {"topic": "AI Research"})
    
    # 搜索
    results = graph_rag.search("nature landscape")
    print(f"Found {len(results)} matching documents")
    
    # 图搜索
    graph_results = graph_rag.graph_search("AI research presentations")
    
    # 问答
    answer = graph_rag.qa_with_multimedia("描述照片中有什麼内容？")
    print(answer)

if __name__ == "__main__":
    example_usage()