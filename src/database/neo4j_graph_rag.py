from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp, HuggingFaceHub
from ..config import *
from ..encoders.multimodal_encoder import MultiModalEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import MultiVectorRetriever
import numpy as np

# Neo4j圖資料庫工具類
class Neo4jGraphRAG:
    def __init__(self, use_local_llm=True):
        self.encoder = MultiModalEncoder()
        
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
                # 如果本地模型加载失败，回退到HuggingFace Hub
                self.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",  # 较小的模型，速度更快
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
        else:
            # 使用HuggingFace Hub上的模型
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-xl",  # 或其他您喜欢的模型
                model_kwargs={"temperature": 0.1, "max_length": 512}
            )
        
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        # 确保Neo4j中存在必要的索引和约束
        self._setup_database()
        
        # 使用HuggingFace Embeddings初始化向量存储
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
                                    `vector.dimensions`: 2048,
                                    `vector.similarity_function`: 'cosine'
                                }}
                            """)
                            print("Neo4j 5.x: 成功创建2048维向量索引")
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
                              2048,
                              'cosine'
                            )
                        """)
                        print("Neo4j 4.x: 成功创建2048维向量索引")
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
                          2048,
                          'cosine'
                        )
                    """)
                except Exception as e2:
                    print(f"尝试创建索引时出错: {e2}")
    
    def add_image(self, image_path, metadata=None):
        """添加图片到图数据库"""
        if metadata is None:
            metadata = {}
        
        # 为图片生成描述
        description = self.encoder.generate_text_description(image_path, self.llm)
        
        # 编码图片
        image_embedding = self.encoder.encode_image(image_path)
        
        # 生成唯一ID
        doc_id = f"img_{os.path.basename(image_path)}_{np.random.randint(10000, 99999)}"
        
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
        
        # 为这个文档在Neo4j中创建节点
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                CREATE (n:MultimediaContent:Image {
                    id: $id,
                    path: $path,
                    type: 'image',
                    description: $description,
                    embedding: $embedding
                })
                RETURN n
            """, {
                "id": doc_id,
                "path": image_path,
                "description": description,
                "embedding": image_embedding.tolist()
            })
            
            # 添加额外的元数据作为属性
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    session.run("""
                        MATCH (n:MultimediaContent {id: $id})
                        SET n.$key = $value
                    """, {"id": doc_id, "key": key, "value": value})
        
        return doc_id
    
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
    
    def search(self, query, k=5):
        """基于文本查询检索相关的多媒体内容"""
        # 使用多向量检索器检索内容
        docs = self.retriever.invoke(query, config={"k": k})
        return docs
    
    def graph_search(self, query, k=5, max_hops=2):
        """使用图结构进行高级检索，考虑节点之间的关系"""
        # 首先获取最相关的节点
        initial_docs = self.search(query, k=k)
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
                RETURN DISTINCT related.id as id, related.type as type, 
                       related.description as description, related.path as path,
                       related.url as url, related.title as title
            """, {
                "ids": initial_ids,
                "max_hops": max_hops
            })
            
            # 处理结果
            related_docs = []
            for record in result:
                # 查找对应的文档
                doc_id = record["id"]
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
    
    def qa_with_multimedia(self, query):
        """基于多媒體内容的問答"""
        # 检索相关内容
        docs = self.graph_search(query, k=3)
        
        # 如果没有找到相關内容
        if not docs:
            return "没有找到相關的多媒體内容来回答您的問題。"
            
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
            if doc_type == "image":
                sources_info.append(f"圖片: {doc.metadata.get('path', 'unknown')}")
            elif doc_type == "video":
                sources_info.append(f"影片: {doc.metadata.get('title', 'unknown')}")
            elif doc_type == "youtube_video":
                sources_info.append(f"YouTube影片: {doc.metadata.get('title', 'unknown')} ({doc.metadata.get('url', 'unknown')})")
        
        # 構建最终回答
        final_answer = f"{answer}\n\n信息来源:\n" + "\n".join(sources_info)
        
        return final_answer