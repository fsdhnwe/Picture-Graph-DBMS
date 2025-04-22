from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.config import *
from src.encoders.multimodal_encoder import MultiModalEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import MultiVectorRetriever
import numpy as np
import os

# Neo4j圖資料庫工具類
class Neo4jGraphRAG:
    def __init__(self, use_local_llm=True, use_mock=False):
        self.encoder = MultiModalEncoder()
        self.use_mock = use_mock
        
        # 初始化LLM - 可以選擇本地模型或HuggingFace Hub上的模型
        if use_local_llm:
            try:
                # 嘗試使用本機LLM（需要預先下載模型）
                self.llm = LlamaCpp(
                    model_path="llama-2-7b-chat.Q3_K_L.gguf",  # 替換為您本地模型的路徑
                    temperature=0.1,
                    max_tokens=2000,
                    n_ctx=4096,
                    verbose=False
                )
            except Exception as e:
                print(f"无法加载本地LLM，将使用HuggingFace模型: {e}")
                # 如果本地模型加载失败，回退到本地HuggingFace模型
                try:
                    # 使用本地HuggingFace模型
                    model_name = "google/flan-t5-base"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    
                    pipe = pipeline(
                        "text2text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_length=512
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                except Exception as e:
                    print(f"使用本地HuggingFace模型失敗: {e}")
                    # 使用簡單文本為備用
                    self.llm = None
        else:
            # 使用HuggingFace Hub上的模型
            try:
                # 使用本地HuggingFace模型
                model_name = "google/flan-t5-large"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                pipe = pipeline(
                    "text2text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_length=512
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                print(f"使用本地HuggingFace模型失敗: {e}")
                # 使用簡單文本為備用
                self.llm = None
        
        if not use_mock:
            try:
                self.driver = GraphDatabase.driver(
                    NEO4J_URI, 
                    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
                )
                
                # 测试连接
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    session.run("RETURN 1")
                    print("成功连接到Neo4j数据库")
                    
                # 确保Neo4j中存在必要的索引和约束
                self._setup_database()
                
                # 使用HuggingFace Embeddings初始化向量存储
                # 注意: 我们现在不需要单独的文本嵌入模型，因为我们使用CLIP进行文本和图像嵌入
                embeddings_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                
                # 初始化向量存储
                try:
                    # 尝试从现有索引创建
                    self.vector_store = Neo4jVector.from_existing_index(
                        embedding=embeddings_model,
                        url=NEO4J_URI,
                        username=NEO4J_USERNAME,
                        password=NEO4J_PASSWORD,
                        database=NEO4J_DATABASE,
                        index_name="multimedia_vector_index",
                        node_label="MultimediaContent",
                        text_node_property="description",
                        embedding_node_property="embedding"
                    )
                except ValueError as e:
                    # 如果维度不匹配，手动创建新的向量存储
                    if "dimensions do not match" in str(e):
                        print("創建新的Neo4j向量存儲")
                        
                        # 使用Neo4jVector的原始构造函数
                        self.vector_store = Neo4jVector(
                            embedding=embeddings_model,
                            url=NEO4J_URI,
                            username=NEO4J_USERNAME,
                            password=NEO4J_PASSWORD,
                            database=NEO4J_DATABASE,
                            index_name="multimedia_vector_index",
                            node_label="MultimediaContent",
                            text_node_property="description",
                            embedding_node_property="embedding"
                        )
                        
                        # 确保索引存在
                        with self.driver.session(database=NEO4J_DATABASE) as session:
                            try:
                                session.run("""
                                    MATCH (n:MultimediaContent)
                                    RETURN count(n) as count
                                """)
                                print("成功连接到Neo4j并验证MultimediaContent节点")
                            except Exception as e:
                                print(f"验证Neo4j连接失败: {e}")
                    else:
                        # 其他错误
                        raise e
                
                # 初始化多向量检索器的存储
                self.doc_store = InMemoryStore()
                
                # 初始化多向量检索器
                self.retriever = MultiVectorRetriever(
                    vectorstore=self.vector_store,
                    docstore=self.doc_store,
                    id_key="doc_id"
                )
            except Exception as e:
                print(f"Neo4j连接失败，切换到模拟模式: {e}")
                self.use_mock = True
                self.doc_store = InMemoryStore()
                self.mock_docs = []
        else:
            print("使用模拟模式，不连接Neo4j数据库")
            self.doc_store = InMemoryStore()
            self.mock_docs = []
    
    def _setup_database(self):
        """设置Neo4j数据库，创建索引和约束"""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            # 创建约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:MultimediaContent) REQUIRE n.id IS UNIQUE")
            
            # 检查Neo4j版本，确定使用哪种API
            try:
                version_result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version").single()
                neo4j_version = version_result["version"] if version_result else "4.0.0"  # 默认假设4.0+
                
                # 将版本字符串转换为数字
                major_version = int(neo4j_version.split(".")[0])
                
                # Neo4j 5.x使用不同的向量索引API
                if major_version >= 5:
                    # 检查索引是否存在
                    index_exists = False
                    try:
                        indexes = session.run("SHOW INDEXES").data()
                        for index in indexes:
                            if "name" in index and index["name"] == "multimedia_vector_index":
                                index_exists = True
                                break
                    except Exception as e:
                        print(f"检查索引时出错: {e}")
                    
                    # 如果索引不存在，创建新索引
                    if not index_exists:
                        try:
                            # Neo4j 5.x的创建向量索引语法
                            session.run("""
                                CREATE VECTOR INDEX multimedia_vector_index 
                                FOR (n:MultimediaContent)
                                ON (n.embedding)
                                OPTIONS {indexConfig: {
                                    `vector.dimensions`: 512,
                                    `vector.similarity_function`: 'cosine'
                                }}
                            """)
                            print("Neo4j 5.x: 成功创建512维向量索引")
                        except Exception as e:
                            print(f"创建Neo4j 5.x向量索引失败: {e}")
                else:
                    # Neo4j 4.x
                    try:
                        session.run("""
                            CALL db.index.vector.createNodeIndex(
                              'multimedia_vector_index',
                              'MultimediaContent',
                              'embedding',
                              512,
                              'cosine'
                            )
                        """)
                        print("Neo4j 4.x: 成功创建512维向量索引")
                    except Exception as e:
                        if "equivalent index already exists" in str(e):
                            print("Neo4j 4.x: 索引已存在")
                        else:
                            print(f"创建Neo4j 4.x向量索引失败: {e}")
                
            except Exception as e:
                print(f"检查Neo4j版本或创建索引时发生错误: {e}")
                # 尝试创建索引，忽略错误
                try:
                    session.run("""
                        CALL db.index.vector.createNodeIndex(
                          'multimedia_vector_index',
                          'MultimediaContent',
                          'embedding',
                          512,
                          'cosine'
                        )
                    """)
                except Exception as e2:
                    print(f"尝试创建索引时出错: {e2}")
    
    def add_image(self, image_path, metadata=None):
        """添加图片到图数据库，如果已存在則跳過"""
        if metadata is None:
            metadata = {}
        
        # 檢查圖片是否已存在於數據庫中
        image_basename = os.path.basename(image_path)
        existing_id = None
        
        if not self.use_mock:
            try:
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    # 查詢數據庫中是否已有相同檔名的圖片
                    result = session.run("""
                        MATCH (n:MultimediaContent:Image)
                        WHERE n.filename = $filename
                        RETURN n.id as id
                    """, {"filename": image_basename})
                    
                    record = result.single()
                    if record:
                        existing_id = record["id"]
                        print(f"圖片 {image_basename} 已存在於數據庫中，ID: {existing_id}")
                        return existing_id
            except Exception as e:
                print(f"檢查圖片存在時發生錯誤: {e}")
                # 繼續添加圖片
        else:
            # 模擬模式下檢查是否已存在
            for doc in self.mock_docs:
                if doc.get("metadata", {}).get("filename") == image_basename:
                    print(f"圖片 {image_basename} 已存在於模擬數據庫中，ID: {doc['id']}")
                    return doc["id"]
        
        try:
            # 為圖片生成描述
            description = "A image containing " + image_basename
            
            # 如果LLM可用，嘗試生成更好的描述
            try:
                llm_description = self.encoder.generate_text_description(image_path, self.llm)
                if llm_description and len(llm_description) > 10:
                    description = llm_description
            except Exception as e:
                print(f"使用LLM生成描述失敗，使用默認描述: {e}")
            
            # 編碼圖片
            image_embedding = self.encoder.encode_image(image_path)
            
            # 生成唯一ID
            doc_id = f"img_{os.path.basename(image_path)}_{np.random.randint(10000, 99999)}"
            
            # 添加檔名到元數據
            metadata["filename"] = image_basename
            
            # 创建文档对象
            doc = Document(
                page_content=description,
                metadata={
                    "doc_id": doc_id,
                    "type": "image",
                    "path": image_path,
                    "description": description,
                    **metadata
                }
            )
            
            # 保存到doc store
            self.doc_store.mset([(doc_id, doc)])
            
            if not self.use_mock:
                try:
                    # 为这个文档在Neo4j中创建节点
                    with self.driver.session(database=NEO4J_DATABASE) as session:
                        # 創建主節點
                        session.run("""
                            CREATE (n:MultimediaContent:Image {
                            id: $id,
                            path: $path,
                            filename: $filename,
                            type: 'image',
                            description: $description,
                            embedding: $embedding
                        })
                        RETURN n
                    """, {
                        "id": doc_id,
                        "path": image_path,
                        "filename": image_basename,
                        "description": description,
                        "embedding": image_embedding.tolist()
                    })
                
                        # 添加額外元數據（安全的方式）
                        if metadata:
                            valid_metadata = {k:v for k,v in metadata.items() 
                                            if isinstance(v, (str, int, float, bool))}
                            if valid_metadata:
                                session.run("""
                                    MATCH (n:MultimediaContent {id: $id})
                                    SET n += $properties
                                """, {"id": doc_id, "properties": valid_metadata})
                    
                    print(f"圖片 {image_basename} 已成功添加到Neo4j數據庫，ID: {doc_id}")
                except Exception as e:
                    print(f"添加圖片到Neo4j時發生錯誤: {e}")
                    # 如果Neo4j存儲失敗，仍然返回文檔ID，因為它已經在內存中
            else:
                # 模拟模式：只存储在内存中
                self.mock_docs.append({
                    "id": doc_id,
                    "type": "image",
                    "path": image_path,
                    "filename": image_basename,
                    "description": description,
                    "embedding": image_embedding,
                    "metadata": metadata
                })
                print(f"圖片 {image_basename} 已成功添加到模擬數據庫，ID: {doc_id}")
            
            return doc_id
        except Exception as e:
            print(f"添加圖片過程中發生未處理的錯誤: {e}")
            if existing_id:
                return existing_id
            return None
    
    def add_video(self, video_path, metadata=None):
        """添加视频到图数据库"""
        if metadata is None:
            metadata = {}
            
        # 编码视频
        video_embedding = self.encoder.encode_video(video_path)
        
        # 视频没有直接的文本描述，可以从元数据或文件名生成
        title = metadata.get("title", os.path.basename(video_path))
        description = metadata.get("description", f"Video file: {title}")
        
        # 生成唯一ID
        doc_id = f"vid_{os.path.basename(video_path)}_{np.random.randint(10000, 99999)}"
        
        # 创建文档对象
        doc = Document(
            page_content=description,
            metadata={
                "doc_id": doc_id,
                "type": "video",
                "path": video_path,
                "title": title,
                "description": description,
                **metadata
            }
        )
        
        # 保存到doc store
        self.doc_store.mset([(doc_id, doc)])
        
        # 为这个文档在Neo4j中创建节点
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                CREATE (n:MultimediaContent:Video {
                    id: $id,
                    path: $path,
                    type: 'video',
                    title: $title,
                    description: $description,
                    embedding: $embedding
                })
                RETURN n
            """, {
                "id": doc_id,
                "path": video_path,
                "title": title,
                "description": description,
                "embedding": video_embedding.tolist()
            })
            
            # 添加额外的元数据作为属性
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    session.run("""
                        MATCH (n:MultimediaContent {id: $id})
                        SET n.$key = $value
                    """, {"id": doc_id, "key": key, "value": value})
        
        return doc_id
    
    def add_youtube_video(self, youtube_url, metadata=None):
        """添加YouTube视频到图数据库"""
        # 下载并编码YouTube视频
        embedding, yt_metadata = self.encoder.encode_youtube_video(youtube_url)
        
        # 合并元数据
        if metadata is None:
            metadata = {}
        metadata = {**yt_metadata, **metadata}
        
        # 生成唯一ID
        video_id = youtube_url.split("v=")[-1] if "v=" in youtube_url else "unknown"
        doc_id = f"yt_{video_id}_{np.random.randint(10000, 99999)}"
        
        # 创建文档对象
        doc = Document(
            page_content=metadata.get("description", "YouTube video"),
            metadata={
                "doc_id": doc_id,
                "type": "youtube_video",
                "url": youtube_url,
                **metadata
            }
        )
        
        # 保存到doc store
        self.doc_store.mset([(doc_id, doc)])
        
        # 为这个文档在Neo4j中创建节点
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                CREATE (n:MultimediaContent:YouTubeVideo {
                    id: $id,
                    url: $url,
                    type: 'youtube_video',
                    title: $title,
                    description: $description,
                    embedding: $embedding
                })
                RETURN n
            """, {
                "id": doc_id,
                "url": youtube_url,
                "title": metadata.get("title", "YouTube Video"),
                "description": metadata.get("description", ""),
                "embedding": embedding.tolist() if embedding is not None else []
            })
        
        return doc_id
    
    def create_relationship(self, source_id, target_id, relationship_type, properties=None):
        """创建两个多媒体内容节点之间的关系"""
        if properties is None:
            properties = {}
            
        if self.use_mock:
            print(f"模拟创建关系: {source_id} -> {relationship_type} -> {target_id}")
            return True
        
        with self.driver.session(database=NEO4J_DATABASE) as session:
            # 创建关系
            result = session.run("""
                MATCH (source:MultimediaContent {id: $source_id})
                MATCH (target:MultimediaContent {id: $target_id})
                CREATE (source)-[r:`$relationship_type` $props]->(target)
                RETURN r
            """, {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "props": properties
            })
            
            return result.single() is not None
    
    def search(self, query, k=5, min_similarity=0.65):
        """基于文本查询检索相关的多媒体内容，使用CLIP进行跨模态检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            min_similarity: 最低相似度阈值，低于此值的结果将被过滤掉
        """
        try:
            # 解析查询意图关键词
            query_keywords = query.lower().split()
            
            # 使用CLIP对查询文本进行编码
            text_embedding = self.encoder.encode_text(query)
            
            if self.use_mock:
                # 模拟模式：在内存中计算余弦相似度
                results = []
                for doc in self.mock_docs:
                    image_embedding = doc["embedding"]
                    # 计算余弦相似度
                    similarity = np.dot(text_embedding, image_embedding) / (
                        np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding))
                    
                    # 检查描述中是否包含查询关键词
                    description = doc.get("description", "").lower()
                    keyword_matches = sum(1 for kw in query_keywords if kw in description)
                    
                    # 根据关键词匹配调整相似度分数
                    adjusted_similarity = similarity
                    if keyword_matches > 0:
                        adjusted_similarity = similarity * (1 + 0.1 * keyword_matches)
                    
                    # 只添加高于阈值的结果
                    if adjusted_similarity >= min_similarity:
                        doc_copy = doc.copy()
                        doc_copy["score"] = float(adjusted_similarity)
                        doc_copy["original_score"] = float(similarity)
                        results.append(doc_copy)
                
                # 按相似度排序
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:k]
                
                # 转换为Document对象
                docs = []
                for result in results:
                    metadata = {
                        "doc_id": result["id"],
                        "type": result["type"],
                        "path": result["path"],
                        "score": result["score"],
                        "original_score": result.get("original_score", result["score"])
                    }
                    metadata.update(result.get("metadata", {}))
                    
                    doc = Document(
                        page_content=result["description"],
                        metadata=metadata
                    )
                    docs.append(doc)
                
                return docs
            
            # 手动构建Cypher查询来查找相似的向量
            try:
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    # 执行向量相似度查询 - 获取更多候选项以便过滤
                    result = session.run("""
                        CALL db.index.vector.queryNodes(
                          'multimedia_vector_index',
                          $expanded_k,
                          $embedding
                        ) YIELD node, score
                        RETURN node.id as id, node.type as type, node.path as path, 
                               node.description as description, node.url as url, 
                               node.title as title, node.filename as filename, score
                        ORDER BY score DESC
                    """, {
                        "expanded_k": k * 3,  # 获取更多候选结果以进行过滤
                        "embedding": text_embedding.tolist()
                    })
                    
                    # 处理结果
                    docs = []
                    records = list(result)
                    
                    for record in records:
                        # 获取原始分数
                        original_score = record["score"]
                        
                        # 如果分数低于阈值，跳过
                        if original_score < min_similarity:
                            continue
                            
                        # 文本相关性分析
                        description = record["description"] or ""
                        # 检查描述中是否包含查询关键词
                        keyword_matches = sum(1 for kw in query_keywords if kw.lower() in description.lower())

                        print(f"Doc: {record['title']} - Original Score: {original_score} - Keyword Matches: {keyword_matches}")

                        
                        # 根据关键词匹配调整相似度分数
                        adjusted_score = original_score
                        if keyword_matches > 0:
                            adjusted_score = original_score * (1 + 0.1 * keyword_matches)
                            
                        doc_id = record["id"]
                        
                        # 计算绝对路径（如果有必要）
                        path = record["path"]
                        
                        # 查找文档存储中的文档
                        stored_doc = self.doc_store.mget([doc_id])[0]
                        
                        # 如果文档存储中没有该文档，则创建一个新的
                        if stored_doc is None:
                            metadata = {
                                "doc_id": doc_id,
                                "type": record["type"],
                                "path": path,
                                "score": adjusted_score,
                                "original_score": original_score,
                                "keyword_matches": keyword_matches
                            }
                            
                            # 根据不同类型添加不同的元数据
                            if record["url"]:
                                metadata["url"] = record["url"]
                            if record["title"]:
                                metadata["title"] = record["title"]
                            if record["filename"]:
                                metadata["filename"] = record["filename"]
                            
                            doc = Document(
                                page_content=description or "No description available",
                                metadata=metadata
                            )
                        else:
                            # 添加相似度分数到现有文档的元数据
                            stored_doc.metadata["score"] = adjusted_score
                            stored_doc.metadata["original_score"] = original_score
                            stored_doc.metadata["keyword_matches"] = keyword_matches
                            doc = stored_doc
                        
                        docs.append(doc)
                    
                    # 按调整后的分数重新排序
                    docs.sort(key=lambda x: x.metadata["score"], reverse=True)
                    return docs[:k]  # 返回top-k结果
            except Exception as e:
                print(f"執行Neo4j向量查詢時發生錯誤: {e}")
                # 如果Neo4j查詢失敗，嘗試從內存中檢索
                print("嘗試從內存中檢索...")
                
                # 從文檔存儲檢索所有文檔
                all_docs = []
                for doc_id in self.doc_store.yield_keys():
                    doc = self.doc_store.mget([doc_id])[0]
                    if doc:
                        all_docs.append(doc)
                
                if not all_docs:
                    print("內存中沒有文檔")
                    return []
                
                # 手動計算相似度
                filtered_docs = []
                for doc in all_docs:
                    # 獲取圖片路徑
                    path = doc.metadata.get("path")
                    if path and os.path.exists(path):
                        try:
                            # 編碼圖片
                            image_embedding = self.encoder.encode_image(path)
                            # 計算余弦相似度
                            similarity = np.dot(text_embedding, image_embedding) / (
                                np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding))
                            
                            # 检查描述中是否包含查询关键词
                            description = doc.page_content.lower()
                            keyword_matches = sum(1 for kw in query_keywords if kw in description)
                            
                            # 根据关键词匹配调整相似度分数
                            adjusted_similarity = similarity
                            if keyword_matches > 0:
                                adjusted_similarity = similarity * (1 + 0.1 * keyword_matches)
                                
                            # 只保留高于阈值的结果
                            if adjusted_similarity >= min_similarity:
                                doc.metadata["score"] = float(adjusted_similarity)
                                doc.metadata["original_score"] = float(similarity)
                                doc.metadata["keyword_matches"] = keyword_matches
                                filtered_docs.append(doc)
                            
                        except Exception as inner_e:
                            print(f"計算圖片 {path} 的相似度時發生錯誤: {inner_e}")
                    
                # 排序並返回結果
                filtered_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
                return filtered_docs[:k]
        
        except Exception as e:
            print(f"Error during CLIP-based search: {e}")
            if not self.use_mock:
                try:
                    # 如果失败，回退到传统的检索方法
                    print("Falling back to traditional retrieval method")
                    docs = self.retriever.invoke(query, config={"k": k})
                    return docs
                except Exception as e2:
                    print(f"传统检索方法也失败了: {e2}")
                    return []
            else:
                # 模拟模式下没有传统检索方法
                print("Error in mock mode search")
                return []
    
    def graph_search(self, query, k=5, max_hops=2, min_similarity=0.65):
        """使用图结构进行高级检索，考虑节点之间的关系
        
        Args:
            query: 查询文本
            k: 返回结果数量
            max_hops: 图遍历的最大跳数
            min_similarity: 最低相似度阈值
        """
        # 首先获取最相关的节点
        initial_docs = self.search(query, k=k, min_similarity=min_similarity)
        
        if self.use_mock:
            # 模拟模式下没有图结构
            return initial_docs
        
        initial_ids = [doc.metadata["doc_id"] for doc in initial_docs]
        
        # 使用Cypher查询进行图扩展
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (n:MultimediaContent)
                WHERE n.id IN $ids
                CALL {
                    WITH n
                    MATCH path = (n)-[*1..$max_hops]-(related:MultimediaContent)
                    RETURN related
                    LIMIT 10
                }
                RETURN DISTINCT related.id as id, related.type as type, related.path as path, 
                       related.description as description, related.url as url, 
                       related.title as title
            """, {
                "ids": initial_ids,
                "max_hops": max_hops
            })
            
            # 处理结果
            related_docs = []
            for record in result:
                doc_id = record["id"]
                
                # 查找文档存储中的文档
                doc = self.doc_store.mget([doc_id])[0]
                if doc:
                    related_docs.append(doc)
            
            # 合并初始文档和关联文档
            all_docs = initial_docs + [doc for doc in related_docs if doc.metadata["doc_id"] not in initial_ids]
            
            return all_docs
    
    def semantic_search_with_filters(self, query, filters=None, k=5):
        """带有過濾條件的語意搜索"""
        if filters is None:
            filters = {}
            
        # 構建Neo4j查詢
        filter_conditions = []
        filter_params = {"query_embedding": []}  # 将在向量存儲中填充
        
        for key, value in filters.items():
            filter_conditions.append(f"n.{key} = ${key}")
            filter_params[key] = value
            
        filter_clause = " AND ".join(filter_conditions) if filter_conditions else ""
        
        # 使用向量存儲進行檢索
        docs = self.vector_store.similarity_search(
            query, 
            k=k,
            filter=filter_clause if filter_clause else None,
            filter_params=filter_params if filter_conditions else None
        )
        
        return docs
    
    def qa_with_multimedia(self, query, min_similarity=0.65):
        """基于多媒體内容的問答
        
        Args:
            query: 查询文本
            min_similarity: 最低相似度阈值
        """
        # 检索相关内容
        docs = self.graph_search(query, k=3, min_similarity=min_similarity)
        
        # 如果没有找到相關内容
        if not docs:
            return "没有找到相關的多媒體内容来回答您的問題。"
            
        if self.use_mock:
            # 模拟模式下的简单实现
            descriptions = [doc.page_content for doc in docs]
            combined_description = " ".join(descriptions)
            
            prompt = f"""根據以下圖片描述，回答問題：
描述：{combined_description}

问题：{query}
"""
            answer = self.llm.invoke(prompt)
            
            # 格式化源文檔信息
            sources_info = []
            for doc in docs:
                doc_type = doc.metadata.get("type", "unknown")
                score = doc.metadata.get("score", 0)
                
                if doc_type == "image":
                    sources_info.append(f"圖片: {doc.metadata.get('path', 'unknown')} (相似度: {score:.4f})")
                elif doc_type == "video":
                    sources_info.append(f"影片: {doc.metadata.get('title', 'unknown')} (相似度: {score:.4f})")
                elif doc_type == "youtube_video":
                    sources_info.append(f"YouTube影片: {doc.metadata.get('title', 'unknown')} ({doc.metadata.get('url', 'unknown')}) (相似度: {score:.4f})")
            
            # 構建最终回答
            final_answer = f"{answer}\n\n信息来源:\n" + "\n".join(sources_info)
            
            return final_answer
        
        # 构建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # 執行處理
        result = qa_chain({"query": query})
        
        # 處理结果
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # 格式化源文檔信息
        sources_info = []
        for doc in source_docs:
            doc_type = doc.metadata.get("type", "unknown")
            score = doc.metadata.get("score", 0)
            
            if doc_type == "image":
                sources_info.append(f"圖片: {doc.metadata.get('path', 'unknown')} (相似度: {score:.4f})")
            elif doc_type == "video":
                sources_info.append(f"影片: {doc.metadata.get('title', 'unknown')} (相似度: {score:.4f})")
            elif doc_type == "youtube_video":
                sources_info.append(f"YouTube影片: {doc.metadata.get('title', 'unknown')} ({doc.metadata.get('url', 'unknown')}) (相似度: {score:.4f})")
        
        # 構建最终回答
        final_answer = f"{answer}\n\n信息来源:\n" + "\n".join(sources_info)
        
        return final_answer