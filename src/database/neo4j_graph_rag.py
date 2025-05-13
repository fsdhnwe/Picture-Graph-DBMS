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
                print(f"加载本地LLM")
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
                print(f"无法加载本地HuggingFace模型，将使用HuggingFace Hub上的模型")
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
    
    def create_or_get_tag(self, tag_name):
        """創建標籤節點，如果已存在則返回現有節點ID"""
        if self.use_mock:
            # 模擬模式
            tag_id = f"tag_{tag_name.lower().replace(' ', '_')}"
            exists = False
            for doc in self.mock_docs:
                if doc.get("type") == "tag" and doc.get("name") == tag_name:
                    exists = True
                    return doc.get("id")
            
            if not exists:
                self.mock_docs.append({
                    "id": tag_id,
                    "type": "tag",
                    "name": tag_name
                })
            return tag_id
        
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # 檢查標籤是否已存在
                result = session.run("""
                    MATCH (t:Tag {name: $name})
                    RETURN t.id as id
                """, {"name": tag_name})
                
                record = result.single()
                if record:
                    # 標籤已存在，返回ID
                    return record["id"]
                
                # 標籤不存在，創建新標籤
                tag_id = f"tag_{tag_name.lower().replace(' ', '_')}"
                session.run("""
                    CREATE (t:Tag {
                        id: $id,
                        name: $name,
                        created_at: datetime()
                    })
                    RETURN t
                """, {
                    "id": tag_id,
                    "name": tag_name
                })
                
                return tag_id
        except Exception as e:
            print(f"創建標籤時發生錯誤: {e}")
            return None
    
    def add_tag_to_image(self, image_id, tag_name):
        """將標籤添加到圖片"""
        # 獲取或創建標籤節點
        tag_id = self.create_or_get_tag(tag_name)
        if not tag_id:
            return False
        
        if self.use_mock:
            print(f"模擬添加標籤: {image_id} -> HAS_TAG -> {tag_id}")
            return True
        
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # 創建圖片與標籤之間的關係
                session.run("""
                    MATCH (i:MultimediaContent:Image {id: $image_id})
                    MATCH (t:Tag {id: $tag_id})
                    MERGE (i)-[r:HAS_TAG]->(t)
                    RETURN r
                """, {
                    "image_id": image_id,
                    "tag_id": tag_id
                })
                
                return True
        except Exception as e:
            print(f"添加標籤到圖片時發生錯誤: {e}")
            return False
    
    def get_all_tags(self):
        """獲取所有標籤"""
        if self.use_mock:
            tags = []
            for doc in self.mock_docs:
                if doc.get("type") == "tag":
                    tags.append({"id": doc.get("id"), "name": doc.get("name")})
            return tags
        
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (t:Tag)
                    RETURN t.id as id, t.name as name
                    ORDER BY t.name
                """)
                
                tags = [{"id": record["id"], "name": record["name"]} for record in result]
                return tags
        except Exception as e:
            print(f"獲取標籤時發生錯誤: {e}")
            return []
    
    def get_image_tags(self, image_id):
        """獲取圖片的所有標籤"""
        if self.use_mock:
            # 模擬模式
            return []
        
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (i:MultimediaContent:Image {id: $image_id})-[:HAS_TAG]->(t:Tag)
                    RETURN t.id as id, t.name as name
                    ORDER BY t.name
                """, {"image_id": image_id})
                
                tags = [{"id": record["id"], "name": record["name"]} for record in result]
                return tags
        except Exception as e:
            print(f"獲取圖片標籤時發生錯誤: {e}")
            return []
    
    def add_image(self, image_path, metadata=None, tags=None):
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
                        
                        # 如果提供了標籤，添加到現有圖片
                        if tags:
                            for tag in tags:
                                self.add_tag_to_image(existing_id, tag)
                        
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
                llm_description = self.encoder.generate_text_description(image_path)
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
                        
                        # 添加標籤
                        if tags:
                            for tag in tags:
                                self.add_tag_to_image(doc_id, tag)
                        
                        # 檢查與其他圖片的相似度，建立關係
                        self._check_and_create_similarity_relations(doc_id, image_embedding)
                    
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
    
    def _check_and_create_similarity_relations(self, image_id, image_embedding):
        """檢查新增圖片與現有圖片的相似度，並創建相似關係，使用APOC進行批次操作"""
        if self.use_mock:
            return
        
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # 使用APOC進行批次向量比較，計算余弦相似度
                result = session.run("""
                    MATCH (new:MultimediaContent:Image {id: $image_id})
                    MATCH (existing:MultimediaContent:Image)
                    WHERE existing.id <> $image_id AND existing.embedding IS NOT NULL
                    WITH new, existing, 
                         apoc.math.cosineSimilarity(new.embedding, existing.embedding) AS similarity
                    WHERE similarity >= 0.75
                    RETURN existing.id AS id, similarity AS score
                    ORDER BY similarity DESC
                    LIMIT 20
                """, {
                    "image_id": image_id
                })
                
                # 收集所有需要創建關係的對
                similar_pairs = [(record["id"], record["score"]) for record in result]
                
                if similar_pairs:
                    # 使用APOC批次創建雙向關係
                    # 注意：APOC的create.relationship會自動處理方向性
                    pairs_param = [
                        {"img1": image_id, "img2": similar_id, "score": similarity}
                        for similar_id, similarity in similar_pairs
                    ]
                    
                    # 執行批次創建關係
                    session.run("""
                        UNWIND $pairs AS pair
                        MATCH (img1:MultimediaContent:Image {id: pair.img1})
                        MATCH (img2:MultimediaContent:Image {id: pair.img2})
                        CALL apoc.create.relationship(img1, 'SIMILAR', {score: pair.score}, img2) YIELD rel as r1
                        CALL apoc.create.relationship(img2, 'SIMILAR', {score: pair.score}, img1) YIELD rel as r2
                        RETURN count(*)
                    """, {"pairs": pairs_param})
                    
                    # 輸出日誌
                    for similar_id, similarity in similar_pairs:
                        print(f"使用APOC創建雙向相似關係: {image_id} <-> SIMILAR({similarity:.4f}) <-> {similar_id}")
                
                print(f"成功使用APOC處理 {len(similar_pairs)} 個相似圖片關係")
                
        except Exception as e:
            print(f"檢查圖片相似度時發生錯誤: {e}")
            # 嘗試退回到基本方法
            print("嘗試使用基本方法創建相似關係...")
            try:
                # 基本方法：直接使用Cypher計算相似度並創建關係
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    # 查找相似的圖片
                    result = session.run("""
                        MATCH (n:MultimediaContent:Image)
                        WHERE n.id <> $image_id
                        WITH n, 
                             reduce(s = 0.0, i IN range(0, size(n.embedding)-1) | 
                                s + n.embedding[i] * $embedding[i]) / 
                             (sqrt(reduce(s = 0.0, x IN n.embedding | s + x * x)) * 
                              sqrt(reduce(s = 0.0, x IN $embedding | s + x * x))) as score
                        WHERE score >= 0.75
                        RETURN n.id as id, score
                        ORDER BY score DESC
                        LIMIT 20
                    """, {
                        "embedding": image_embedding.tolist(),
                        "image_id": image_id
                    })
                    
                    # 創建相似關係
                    for record in result:
                        similar_id = record["id"]
                        similarity = record["score"]
                        
                        # 創建雙向相似關係
                        session.run("""
                            MATCH (img1:MultimediaContent:Image {id: $img1_id})
                            MATCH (img2:MultimediaContent:Image {id: $img2_id})
                            MERGE (img1)-[r1:SIMILAR {score: $score}]->(img2)
                            MERGE (img2)-[r2:SIMILAR {score: $score}]->(img1)
                            RETURN r1, r2
                        """, {
                            "img1_id": image_id,
                            "img2_id": similar_id,
                            "score": similarity
                        })
                        
                        print(f"創建雙向相似關係: {image_id} <-> SIMILAR({similarity:.4f}) <-> {similar_id}")
            except Exception as e2:
                print(f"基本方法也失敗了: {e2}")
    
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
    
    def search(self, query, k=5, min_similarity=0.7):
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
                        MATCH (node:MultimediaContent:Image)
                        WHERE NOT node.id IN $existing_ids
                        WITH node, 
                             reduce(s = 0.0, i IN range(0, size(node.embedding)-1) | 
                                s + node.embedding[i] * $embedding[i]) / 
                             (sqrt(reduce(s = 0.0, x IN node.embedding | s + x * x)) * 
                              sqrt(reduce(s = 0.0, x IN $embedding | s + x * x))) as score
                        WHERE score >= $min_similarity
                        
                        // 獲取標籤匹配計數
                        OPTIONAL MATCH (node)-[:HAS_TAG]->(t:Tag)
                        WHERE t.id IN $tag_ids
                        
                        WITH node, score, count(DISTINCT t) as tag_match_count
                        
                        RETURN node.id as id, node.path as path, 
                               node.description as description, 
                               node.filename as filename,
                               'VECTOR_MATCH' as match_type,
                               tag_match_count,
                               score as initial_score
                        ORDER BY tag_match_count DESC, score DESC
                        LIMIT 20
                    """, {
                        "embedding": text_embedding.tolist(),
                        "existing_ids": [img["id"] for img in results],
                        "tag_ids": [tag["id"] for tag in results if tag.get("type") == "tag"],
                        "min_similarity": min_similarity
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
    
    def graph_search(self, query, k=5, max_hops=2, min_similarity=0.7):
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
    
    def qa_with_multimedia(self, query, min_similarity=0.7):
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

    def get_all_images(self, limit=50):
        """獲取數據庫中所有的圖片"""
        if self.use_mock:
            return self.mock_docs[:limit] if self.mock_docs else []
        
        try:
            # 使用Cypher查詢獲取圖片內容
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (n:MultimediaContent)
                    WHERE n.type = 'image' OR n.type = 'Image'
                    RETURN n.id as id, n.description as description, 
                           n.path as path, n.filename as filename,
                           n.type as type, n.embedding as embedding
                    LIMIT $limit
                """, limit=limit).data()
            
            # 將結果轉換為Document對象
            docs = []
            for item in result:
                # 創建Document對象
                doc = Document(
                    page_content=item.get("description", ""),
                    metadata={
                        "doc_id": item.get("id"),
                        "path": item.get("path"),
                        "filename": item.get("filename"),
                        "type": item.get("type"),
                        "embedding": np.array(item["embedding"]) if item.get("embedding") is not None else None 
                    }
                )
                docs.append(doc)
            
            return docs
        except Exception as e:
            print(f"獲取圖片時發生錯誤: {e}")
            return []
            
    def advanced_search(self, query, k=10, min_similarity=0.7):
        """進階搜尋功能，結合標籤匹配、圖探索和向量相似度
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            min_similarity: 最低相似度閾值
            
        Returns:
            List of Document objects with search metadata
        """
        if self.use_mock:
            # 模擬模式下使用簡單搜尋
            return self.search(query, k=k, min_similarity=min_similarity)
            
        # 解析查詢詞
        query_words = [word.lower().strip() for word in query.split() if len(word.strip()) > 1]
        
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # 步驟 1: 標籤匹配
                print(f"步驟 1: 查詢標籤匹配 - 關鍵詞: {query_words}")
                tags_result = session.run("""
                    // 找出與查詢詞匹配的標籤
                    MATCH (t:Tag)
                    WHERE any(word IN $query_words WHERE toLower(t.name) CONTAINS word)
                    RETURN t.id as tag_id, t.name as tag_name
                """, {"query_words": query_words})
                
                matching_tags = [(record["tag_id"], record["tag_name"]) for record in tags_result]
                print(f"找到匹配的標籤: {matching_tags}")
                
                # 如果找到匹配的標籤，查詢具有這些標籤的圖片
                candidate_images = []
                if matching_tags:
                    tag_ids = [tag_id for tag_id, _ in matching_tags]
                    images_with_tags_result = session.run("""
                        // 查詢具有匹配標籤的圖片，並計算標籤匹配數量
                        MATCH (img:MultimediaContent:Image)-[:HAS_TAG]->(t:Tag)
                        WHERE t.id IN $tag_ids
                        WITH img, count(DISTINCT t) as tag_match_count
                        RETURN img.id as id, img.path as path, img.description as description, 
                               img.filename as filename, 
                               'TAG_MATCH' as match_type,
                               tag_match_count,
                               1.0 as initial_score
                        ORDER BY tag_match_count DESC
                    """, {"tag_ids": tag_ids})
                    
                    for record in images_with_tags_result:
                        candidate_images.append({
                            "id": record["id"],
                            "path": record["path"],
                            "description": record["description"],
                            "filename": record["filename"],
                            "match_type": record["match_type"],
                            "tag_match_count": record["tag_match_count"],
                            "score": record["initial_score"]
                        })
                    
                    print(f"步驟 1: 找到 {len(candidate_images)} 個標籤匹配的圖片")
                
                # 步驟 2: 通過相似圖片關係擴展結果
                if candidate_images:
                    print("步驟 2: 擴展通過相似關係")
                    seed_image_ids = [img["id"] for img in candidate_images]
                    similar_images_result = session.run("""
                        // 通過SIMILAR關係擴展結果
                        MATCH (seed:MultimediaContent:Image)-[sim:SIMILAR]->(related:MultimediaContent:Image)
                        WHERE seed.id IN $seed_ids
                          AND sim.score >= $min_similarity
                          AND NOT related.id IN $seed_ids
                        
                        // 取得相關圖片的標籤匹配計數
                        OPTIONAL MATCH (related)-[:HAS_TAG]->(t:Tag)
                        WHERE t.id IN $tag_ids
                        
                        WITH related, sim.score as similarity, count(DISTINCT t) as tag_match_count
                        
                        RETURN related.id as id, related.path as path, 
                               related.description as description, 
                               related.filename as filename,
                               'SIMILAR_EXPAND' as match_type,
                               tag_match_count,
                               similarity as initial_score
                        ORDER BY tag_match_count DESC, similarity DESC
                    """, {
                        "seed_ids": seed_image_ids,
                        "tag_ids": tag_ids if matching_tags else [],
                        "min_similarity": min_similarity
                    })
                    
                    similar_images = []
                    for record in similar_images_result:
                        similar_images.append({
                            "id": record["id"],
                            "path": record["path"],
                            "description": record["description"],
                            "filename": record["filename"],
                            "match_type": record["match_type"],
                            "tag_match_count": record["tag_match_count"],
                            "score": record["initial_score"]
                        })
                    
                    print(f"步驟 2: 找到 {len(similar_images)} 個相似相關的圖片")
                    candidate_images.extend(similar_images)
                
                # 步驟 3: 如果標籤匹配結果少，使用CLIP向量相似度
                if len(candidate_images) < 10 or len(matching_tags) <= 1:
                    print("步驟 3: 使用CLIP向量相似度進行查詢")
                    # 使用CLIP對查詢文本進行編碼
                    text_embedding = self.encoder.encode_text(query)
                    
                    # 查詢向量相似的圖片
                    vector_query_result = session.run("""
                        // 向量相似度查詢
                        MATCH (node:MultimediaContent:Image)
                        WHERE NOT node.id IN $existing_ids
                        WITH node, 
                             reduce(s = 0.0, i IN range(0, size(node.embedding)-1) | 
                                s + node.embedding[i] * $embedding[i]) / 
                             (sqrt(reduce(s = 0.0, x IN node.embedding | s + x * x)) * 
                              sqrt(reduce(s = 0.0, x IN $embedding | s + x * x))) as score
                        WHERE score >= $min_similarity
                        
                        // 獲取標籤匹配計數
                        OPTIONAL MATCH (node)-[:HAS_TAG]->(t:Tag)
                        WHERE t.id IN $tag_ids
                        
                        WITH node, score, count(DISTINCT t) as tag_match_count
                        
                        RETURN node.id as id, node.path as path, 
                               node.description as description, 
                               node.filename as filename,
                               'VECTOR_MATCH' as match_type,
                               tag_match_count,
                               score as initial_score
                        ORDER BY tag_match_count DESC, score DESC
                        LIMIT 20
                    """, {
                        "embedding": text_embedding.tolist(),
                        "existing_ids": [img["id"] for img in candidate_images],
                        "tag_ids": tag_ids if matching_tags else [],
                        "min_similarity": min_similarity
                    })
                    
                    vector_matches = []
                    for record in vector_query_result:
                        vector_matches.append({
                            "id": record["id"],
                            "path": record["path"],
                            "description": record["description"],
                            "filename": record["filename"],
                            "match_type": record["match_type"],
                            "tag_match_count": record["tag_match_count"],
                            "score": record["initial_score"]
                        })
                    
                    print(f"步驟 3: 找到 {len(vector_matches)} 個向量相似的圖片")
                    candidate_images.extend(vector_matches)
                
                # 步驟 4: 如果結果仍然不足，使用相關標籤擴展
                if len(candidate_images) < 10 and matching_tags:
                    print("步驟 4: 使用相關標籤擴展結果")
                    
                    related_tags_result = session.run("""
                        // 查詢與已匹配標籤相關的標籤
                        MATCH (t:Tag)-[:RELATED_TO]-(related_tag:Tag)
                        WHERE t.id IN $tag_ids
                          AND NOT related_tag.id IN $tag_ids
                        RETURN DISTINCT related_tag.id as tag_id, related_tag.name as tag_name
                    """, {"tag_ids": tag_ids})
                    
                    related_tag_ids = [record["tag_id"] for record in related_tags_result]
                    print(f"找到相關標籤: {related_tag_ids}")
                    
                    if related_tag_ids:
                        related_tag_images_result = session.run("""
                            // 查詢具有相關標籤的圖片
                            MATCH (img:MultimediaContent:Image)-[:HAS_TAG]->(t:Tag)
                            WHERE t.id IN $related_tag_ids
                              AND NOT img.id IN $existing_ids
                            
                            WITH img, count(DISTINCT t) as related_tag_count
                            
                            RETURN img.id as id, img.path as path, 
                                   img.description as description, 
                                   img.filename as filename,
                                   'RELATED_TAG' as match_type,
                                   0 as tag_match_count,
                                   0.7 as initial_score
                            ORDER BY related_tag_count DESC
                            LIMIT 10
                        """, {
                            "related_tag_ids": related_tag_ids,
                            "existing_ids": [img["id"] for img in candidate_images]
                        })
                        
                        related_tag_images = []
                        for record in related_tag_images_result:
                            related_tag_images.append({
                                "id": record["id"],
                                "path": record["path"],
                                "description": record["description"],
                                "filename": record["filename"],
                                "match_type": record["match_type"],
                                "tag_match_count": record["tag_match_count"],
                                "score": record["initial_score"]
                            })
                        
                        print(f"步驟 4: 找到 {len(related_tag_images)} 個相關標籤的圖片")
                        candidate_images.extend(related_tag_images)
                
                # 步驟 5: 計算描述文本與查詢的相似度，調整分數
                print("步驟 5: 根據描述相似度調整分數")
                
                for image in candidate_images:
                    description = image.get("description", "").lower()
                    if not description:
                        continue
                        
                    # 計算關鍵詞匹配
                    keyword_matches = sum(1 for word in query_words if word in description)
                    
                    # 調整分數
                    if keyword_matches > 0:
                        # 關鍵詞匹配分數加成 (最多加成 0.1)
                        keyword_bonus = min(0.1, 0.02 * keyword_matches)
                        image["score"] += keyword_bonus
                        image["description_match"] = True
                        image["keyword_matches"] = keyword_matches
                
                # 步驟 6: 最終結果排序與組裝
                print("步驟 6: 組合最終結果")
                
                # 按標籤匹配數和分數排序
                sorted_candidates = sorted(
                    candidate_images, 
                    key=lambda x: (x.get("tag_match_count", 0), x.get("score", 0)), 
                    reverse=True
                )
                
                # 限制結果數量
                top_results = sorted_candidates[:k]
                
                # 轉換為Document對象
                result_docs = []
                for result in top_results:
                    doc_id = result["id"]
                    
                    # 查找文檔存儲中是否有該文檔
                    stored_doc = None
                    try:
                        stored_doc = self.doc_store.mget([doc_id])[0]
                    except:
                        pass
                    
                    if stored_doc:
                        # 更新現有文檔的元數據
                        stored_doc.metadata.update({
                            "score": result["score"],
                            "match_type": result["match_type"],
                            "tag_match_count": result["tag_match_count"],
                            "description_match": result.get("description_match", False),
                            "keyword_matches": result.get("keyword_matches", 0)
                        })
                        result_docs.append(stored_doc)
                    else:
                        # 創建新的Document對象
                        metadata = {
                            "doc_id": doc_id,
                            "type": "image",
                            "path": result["path"],
                            "filename": result["filename"],
                            "score": result["score"],
                            "match_type": result["match_type"],
                            "tag_match_count": result["tag_match_count"],
                            "description_match": result.get("description_match", False),
                            "keyword_matches": result.get("keyword_matches", 0)
                        }
                        
                        doc = Document(
                            page_content=result["description"] or f"Image: {result['filename']}",
                            metadata=metadata
                        )
                        result_docs.append(doc)
                
                print(f"最終找到 {len(result_docs)} 個結果")
                return result_docs
        
        except Exception as e:
            print(f"進階搜尋執行時發生錯誤: {e}")
            # 回退到基本搜尋
            print("回退到基本搜尋方法")
            return self.search(query, k=k, min_similarity=min_similarity)

    def __del__(self):
        """清理連接"""