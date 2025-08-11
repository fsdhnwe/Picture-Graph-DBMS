import logging
import time
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain_community.llms import LlamaCpp
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.config import *
from src.encoders.multimodal_encoder import MultiModalEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

class ClipMilvus:
    def __init__(self, use_local_llm=True, use_mock=False, milvus_config=None):
        self.encoder = MultiModalEncoder()
        self.use_mock = use_mock
        
        # 初始化LLM - 可以選擇本地模型或HuggingFace Hub上的模型
        if use_local_llm:
            try:
                print(f"加載本地LLM")
                # 嘗試使用本機LLM（需要預先下載模型）
                self.llm = LlamaCpp(
                    model_path="llama-2-7b-chat.Q3_K_L.gguf",  # 替換為您本地模型的路徑
                    temperature=0.1,
                    max_tokens=2000,
                    n_ctx=4096,
                    verbose=False
                )
            except Exception as e:
                print(f"無法加載本地LLM，將使用HuggingFace模型: {e}")
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
                print(f"無法加載本地HuggingFace模型，將使用HuggingFace Hub上的模型")
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
                self.milvus_config = milvus_config
                self.collection_name = "multimedia_content"  # Milvus collection 名稱
                self.cluster_collection_name = "cluster_centroids"  # 聚類中心點 collection 名稱
                
                # 聚類相關參數
                self.similarity_threshold = 0.8  # 相似度閾值，用於決定是否加入現有聚類
                self.max_cluster_size = 50  # 每個聚類的最大圖片數
                
                # 設置日誌
                logging.basicConfig(level=logging.INFO)
                self.logger = logging.getLogger(__name__)
                
                # 初始化 Milvus 連接
                try:
                    connections.connect(
                        alias="default",
                        host=milvus_config['host'],
                        port=milvus_config['port']
                    )
                    self.logger.info("Successfully connected to Milvus")
                    
                    # 檢查連接是否成功
                    try:
                        # 嘗試列出所有集合，確認連接正常
                        all_collections = utility.list_collections()
                        self.logger.info(f"現有集合: {all_collections}")
                    except Exception as list_error:
                        self.logger.error(f"列出集合時發生錯誤: {list_error}")
                        raise list_error
                        
                except Exception as e:
                    self.logger.error(f"Failed to connect to Milvus: {e}")
                    raise e
                
                # 使用HuggingFace Embeddings初始化向量存储
                # 注意: 我们现在不需要单独的文本嵌入模型，因为我们使用CLIP进行文本和图像嵌入
                embeddings_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                
                # 創建 Milvus collection (文章向量)
                try:
                    self._create_collection()
                    # 創建聚類中心點 collection
                    self._create_cluster_collection()
                    
                
                except Exception as create_error:
                    self.logger.error(f"創建集合時發生錯誤: {create_error}")
                    raise create_error
                
                # 初始化多向量检索器的存储
                self.doc_store = InMemoryStore()
                
            except Exception as e:
                print(f"milvus連接失敗，切換到模擬模式: {e}")
                self.use_mock = True
                self.doc_store = InMemoryStore()
                self.mock_docs = []
        else:
            print("使用模擬模式，不連接Neo4j數據庫")
            self.doc_store = InMemoryStore()
            self.mock_docs = []

    # 初始化向量存储
    def _create_collection(self):
        """
        创建 Milvus collection 如果不存在
        """
        try:
            # 檢查集合是否已存在
            if utility.has_collection(self.collection_name):
                self.logger.info(f"Collection {self.collection_name} 已存在")
                return
            
            # 如果集合不存在，創建新集合
            self.logger.info(f"創建新集合 {self.collection_name}")
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="cluster_id", dtype=DataType.INT64),
                FieldSchema(name="created_time", dtype=DataType.INT64),
                FieldSchema(name="updated_time", dtype=DataType.INT64),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
            ]
            
            schema = CollectionSchema(fields, "picture collection schema")
            collection = Collection(self.collection_name, schema)
            
            # 創建集合後等待一下
            self.logger.info("等待集合創建完成...")
            time.sleep(2)
            
            # 確認集合已創建
            if utility.has_collection(self.collection_name):
                self.logger.info(f"集合 {self.collection_name} 已成功創建")
                
                # 創建索引
                self._create_index(collection)
                
                # 創建成功訊息
                self.logger.info(f"成功創建集合 {self.collection_name} 並添加索引")
            else:
                self.logger.error(f"集合 {self.collection_name} 創建失敗")
        except Exception as e:
            self.logger.error(f"創建集合時發生錯誤: {e}")
            raise e
        
    def _create_index(self, collection):
        """
        為集合的 embedding 欄位創建向量索引
        """
        try:
            # 為 embedding 欄位創建索引
            index_params = {
                "metric_type": "COSINE",  # 使用餘弦相似度
                "index_type": "IVF_FLAT",  # 選擇合適的索引類型
                "params": {"nlist": 128}   # 減少 nlist 數量
            }
            
            # 創建索引前先檢查是否已存在
            try:
                # 檢查是否已有索引
                if collection.has_index():
                    # 嘗試刪除已有的索引
                    collection.drop_index()
                    self.logger.info(f"已刪除集合 {self.collection_name} 上的現有索引")
            except Exception as drop_error:
                self.logger.info(f"處理索引時發生錯誤: {drop_error}")
            
            # 創建索引
            try:
                collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                
                self.logger.info(f"為集合 {self.collection_name} 的 embedding 欄位創建索引成功")
            except Exception as create_index_error:
                self.logger.error(f"創建索引時發生錯誤: {create_index_error}")
                raise create_index_error
            
            # 等待一下讓索引生效
            time.sleep(2)
            
            # 加載集合以使索引生效
            try:
                collection.load()
                self.logger.info(f"集合 {self.collection_name} 加載成功")
            except Exception as load_error:
                self.logger.error(f"加載集合時發生錯誤: {load_error}")
                
        except Exception as e:
            self.logger.error(f"創建索引時發生錯誤: {e}")
            raise e
    
    def _create_cluster_collection(self):
        """
        創建聚類中心點 Milvus collection 如果不存在
        """
        try:
            # 檢查聚類集合是否已存在
            if utility.has_collection(self.cluster_collection_name):
                self.logger.info(f"Cluster collection {self.cluster_collection_name} 已存在")
                return
            
            # 如果集合不存在，創建新集合
            self.logger.info(f"創建新聚類集合 {self.cluster_collection_name}")
            
            # 定义聚類中心點字段
            cluster_fields = [
                FieldSchema(name="cluster_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="centroid_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="count", dtype=DataType.INT64),
                FieldSchema(name="created_time", dtype=DataType.INT64),
                FieldSchema(name="updated_time", dtype=DataType.INT64),
            ]
            
            cluster_schema = CollectionSchema(cluster_fields, "cluster centroids collection schema")
            cluster_collection = Collection(self.cluster_collection_name, cluster_schema)
            
            # 創建集合後等待一下
            self.logger.info("等待聚類集合創建完成...")
            time.sleep(2)
            
            # 確認集合已創建
            if utility.has_collection(self.cluster_collection_name):
                self.logger.info(f"聚類集合 {self.cluster_collection_name} 已成功創建")
                
                # 創建 HNSW 索引
                self._create_cluster_index(cluster_collection)
                
                # 創建成功訊息
                self.logger.info(f"成功創建聚類集合 {self.cluster_collection_name} 並添加 HNSW 索引")
            else:
                self.logger.error(f"聚類集合 {self.cluster_collection_name} 創建失敗")
        except Exception as e:
            self.logger.error(f"創建聚類集合時發生錯誤: {e}")
            raise e
    
    def _create_cluster_index(self, collection):
        """
        為聚類集合的 centroid_vector 欄位創建 HNSW 索引
        """
        try:
            # 為 centroid_vector 欄位創建 HNSW 索引
            index_params = {
                "metric_type": "COSINE",  # 使用餘弦相似度
                "index_type": "HNSW",     # 使用 HNSW 索引
                "params": {
                    "M": 16,              # HNSW 參數：每個節點的最大連接數
                    "efConstruction": 200  # HNSW 參數：構建時的搜索範圍
                }
            }
            
            # 創建索引前先檢查是否已存在
            try:
                if collection.has_index():
                    collection.drop_index()
                    self.logger.info(f"已刪除聚類集合 {self.cluster_collection_name} 上的現有索引")
            except Exception as drop_error:
                self.logger.info(f"處理聚類索引時發生錯誤: {drop_error}")
            
            # 創建 HNSW 索引
            try:
                collection.create_index(
                    field_name="centroid_vector",
                    index_params=index_params
                )
                
                self.logger.info(f"為聚類集合 {self.cluster_collection_name} 的 centroid_vector 欄位創建 HNSW 索引成功")
            except Exception as create_index_error:
                self.logger.error(f"創建聚類索引時發生錯誤: {create_index_error}")
                raise create_index_error
            
            # 等待一下讓索引生效
            time.sleep(2)
            
            # 加載集合以使索引生效
            try:
                collection.load()
                self.logger.info(f"聚類集合 {self.cluster_collection_name} 加載成功")
            except Exception as load_error:
                self.logger.error(f"加載聚類集合時發生錯誤: {load_error}")
                
        except Exception as e:
            self.logger.error(f"創建聚類索引時發生錯誤: {e}")
            raise e
    
    def _find_or_create_cluster(self, image_embedding):
        """
        為圖片嵌入向量尋找合適的聚類，或創建新聚類
        
        Args:
            image_embedding: 圖片的嵌入向量
            
        Returns:
            cluster_id: 聚類ID
        """
        try:
            # 獲取聚類集合
            cluster_collection = Collection(self.cluster_collection_name)
            
            try:
                cluster_collection.load()
                
                # 使用 HNSW 搜索最相似的聚類中心點
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": 64}  # HNSW 搜索參數
                }
                
                # 搜索最相似的聚類
                results = cluster_collection.search(
                    data=[image_embedding.tolist()],
                    anns_field="centroid_vector",
                    param=search_params,
                    limit=1,  # 只找最相似的一個聚類
                    output_fields=["cluster_id", "count"]
                )
                
                # 檢查是否找到相似的聚類
                if results and len(results) > 0 and len(results[0]) > 0:
                    hit = results[0][0]
                    similarity = hit.score
                    cluster_id = hit.entity.get("cluster_id")
                    current_count = hit.entity.get("count")
                    
                    # 如果相似度超過閾值且聚類未滿，則加入現有聚類
                    if similarity >= self.similarity_threshold and current_count < self.max_cluster_size:
                        self.logger.info(f"找到相似聚類 {cluster_id}，相似度: {similarity:.4f}")
                        
                        # 更新聚類中心點和計數
                        self._update_cluster_centroid(cluster_id, image_embedding, current_count)
                        return cluster_id
                
                # 如果沒有找到合適的聚類，創建新聚類
                return self._create_new_cluster(image_embedding)
                
            except Exception as search_error:
                self.logger.error(f"搜索聚類時發生錯誤: {search_error}")
                # 如果搜索失敗，創建新聚類
                return self._create_new_cluster(image_embedding)
            
            finally:
                try:
                    cluster_collection.release()
                except Exception as release_error:
                    self.logger.error(f"釋放聚類集合時發生錯誤: {release_error}")
                    
        except Exception as e:
            self.logger.error(f"尋找或創建聚類時發生錯誤: {e}")
            # 發生錯誤時創建新聚類
            return self._create_new_cluster(image_embedding)
    
    def _create_new_cluster(self, image_embedding):
        """
        創建新的聚類中心點
        
        Args:
            image_embedding: 圖片的嵌入向量
            
        Returns:
            cluster_id: 新聚類的ID
        """
        try:
            # 生成新的聚類ID
            cluster_id = np.random.randint(100000, 9999999)
            current_time = int(time.time())
            
            # 準備插入數據
            insert_data = [
                [cluster_id],                      # cluster_id
                [image_embedding.tolist()],        # centroid_vector
                [1],                               # count (初始為1)
                [current_time],                    # created_time
                [current_time]                     # updated_time
            ]
            
            # 插入新聚類中心點
            cluster_collection = Collection(self.cluster_collection_name)
            cluster_collection.insert(insert_data)
            cluster_collection.flush()
            cluster_collection.release()
            
            self.logger.info(f"創建新聚類 {cluster_id}")
            return cluster_id
            
        except Exception as e:
            self.logger.error(f"創建新聚類時發生錯誤: {e}")
            # 如果創建失敗，返回默認聚類ID
            return 0
    
    def _update_cluster_centroid(self, cluster_id, new_embedding, current_count):
        """
        更新聚類中心點（使用增量平均法）
        
        Args:
            cluster_id: 聚類ID
            new_embedding: 新圖片的嵌入向量
            current_count: 當前聚類中的圖片數量
        """
        try:
            cluster_collection = Collection(self.cluster_collection_name)
            cluster_collection.load()
            
            # 獲取當前聚類中心點
            results = cluster_collection.query(
                expr=f"cluster_id == {cluster_id}",
                output_fields=["centroid_vector"]
            )
            
            if results and len(results) > 0:
                current_centroid = np.array(results[0]["centroid_vector"])
                
                # 使用增量平均法更新中心點
                # new_centroid = (current_centroid * current_count + new_embedding) / (current_count + 1)
                new_centroid = (current_centroid * current_count + new_embedding) / (current_count + 1)
                new_count = current_count + 1
                current_time = int(time.time())
                
                # 刪除舊記錄
                cluster_collection.delete(f"cluster_id == {cluster_id}")
                cluster_collection.flush()
                
                # 插入更新後的記錄
                insert_data = [
                    [cluster_id],                    # cluster_id
                    [new_centroid.tolist()],         # centroid_vector
                    [new_count],                     # count
                    [results[0].get("created_time", current_time)],  # created_time (保持原值)
                    [current_time]                   # updated_time
                ]
                
                cluster_collection.insert(insert_data)
                cluster_collection.flush()
                
                self.logger.info(f"更新聚類 {cluster_id} 中心點，新計數: {new_count}")
            
            cluster_collection.release()
            
        except Exception as e:
            self.logger.error(f"更新聚類中心點時發生錯誤: {e}")
    

    def add_image(self, image_path, metadata=None, tags=None):
        """添加图片到向量数据库，如果已存在則跳過"""
        if metadata is None:
            metadata = {}
        
        # 檢查圖片是否已存在於數據庫中
        image_basename = os.path.basename(image_path)
        existing_id = None
        
        if not self.use_mock:
            try:
                # 在 Milvus 中查詢是否已有相同檔名的圖片
                try:
                    # 獲取集合
                    collection = Collection(self.collection_name)
                    
                    # 嘗試加載集合
                    try:
                        collection.load()
                        
                        # 使用 query 而不是 search，因為我們是精確查詢而不是相似度查詢
                        expr = f"filename == '{image_basename}'"
                        results = collection.query(
                            expr=expr,
                            output_fields=["id", "path", "filename"]
                        )
                        
                        if results and len(results) > 0:
                            existing_id = str(results[0]["id"])
                            print(f"圖片 {image_basename} 已存在於 Milvus 數據庫中，ID: {existing_id}")
                            collection.release()
                            return existing_id
                            
                    except Exception as load_error:
                        print(f"加載集合時發生錯誤: {load_error}")
                        # 嘗試重新建立索引
                        try:
                            self._create_index(collection)
                            collection.load()
                            
                            # 再次嘗試查詢
                            expr = f"filename == '{image_basename}'"
                            results = collection.query(
                                expr=expr,
                                output_fields=["id", "path", "filename"]
                            )
                            
                            if results and len(results) > 0:
                                existing_id = str(results[0]["id"])
                                print(f"圖片 {image_basename} 已存在於 Milvus 數據庫中，ID: {existing_id}")
                                collection.release()
                                return existing_id
                                
                        except Exception as index_error:
                            print(f"重建索引後查詢時發生錯誤: {index_error}")
                            # 如果查詢失敗，繼續添加新圖片
                    
                    finally:
                        # 確保釋放資源
                        try:
                            collection.release()
                        except Exception as release_error:
                            print(f"釋放集合時發生錯誤: {release_error}")
                
                except Exception as collection_error:
                    print(f"獲取集合時發生錯誤: {collection_error}")
                    # 嘗試重新創建集合
                    try:
                        self._create_collection()
                        # 新創建的集合中肯定沒有這個圖片，所以繼續執行
                    except Exception as recreate_error:
                        print(f"重新創建集合時發生錯誤: {recreate_error}")
            
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
            
            # 尋找或創建聚類
            cluster_id = 0  # 默認聚類ID
            if not self.use_mock:
                try:
                    cluster_id = self._find_or_create_cluster(image_embedding)
                except Exception as cluster_error:
                    self.logger.error(f"處理聚類時發生錯誤: {cluster_error}")
                    cluster_id = 0  # 使用默認聚類ID
            
            # 生成唯一ID
            numeric_id = np.random.randint(10000, 9999999)
            doc_id = f"img_{os.path.basename(image_path)}_{numeric_id}"
            
            # 添加檔名和聚類ID到元數據
            metadata["filename"] = image_basename
            metadata["cluster_id"] = cluster_id
            
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
                    # 準備插入到 Milvus 的數據
                    current_time = int(time.time())
                    
                    # 獲取向量維度，確保與 schema 一致
                    embedding_list = image_embedding.tolist()
                    
                    # 創建符合 Milvus 要求的數據格式
                    insert_data = [
                        [numeric_id],  # id 必須是 INT64
                        [embedding_list],  # embedding 是向量
                        [cluster_id],  # cluster_id 是 INT64
                        [current_time],  # created_time 是 INT64
                        [current_time],  # updated_time 是 INT64
                        [str(description)],  # description 是 VARCHAR
                        [str(image_basename)],  # filename 是 VARCHAR
                        [str(image_path)],  # path 是 VARCHAR
                        ["image"]  # type 是 VARCHAR
                    ]
                    
                    # 向 Milvus 插入數據
                    try:
                        collection = Collection(self.collection_name)
                        # 檢查集合是否已加載
                        try:
                            # 嘗試插入數據
                            collection.insert(insert_data)
                            collection.flush()
                            print(f"圖片 {image_basename} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                        except Exception as insert_error:
                            print(f"插入數據時發生錯誤: {insert_error}")
                            # 嘗試重新建立索引
                            try:
                                self._create_index(collection)
                                # 再次嘗試插入
                                collection.insert(insert_data)
                                collection.flush()
                                print(f"重新建立索引後，圖片 {image_basename} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                            except Exception as retry_error:
                                print(f"重新嘗試插入時發生錯誤: {retry_error}")
                                raise retry_error
                        finally:
                            # 確保在所有情況下都釋放資源
                            try:
                                collection.release()
                            except Exception as release_error:
                                print(f"釋放集合時發生錯誤: {release_error}")
                    except Exception as collection_error:
                        print(f"獲取集合時發生錯誤: {collection_error}")
                        # 嘗試重新創建集合
                        try:
                            self._create_collection()
                            # 獲取新創建的集合
                            collection = Collection(self.collection_name)
                            collection.insert(insert_data)
                            collection.flush()
                            collection.release()
                            print(f"重新創建集合後，圖片 {image_basename} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                        except Exception as recreate_error:
                            print(f"重新創建集合後插入數據時發生錯誤: {recreate_error}")
                            raise recreate_error
                except Exception as e:
                    print(f"添加圖片到 Milvus 時發生錯誤: {e}")
                    # 如果 Milvus 存儲失敗，仍然返回文檔ID，因為它已經在內存中
            else:
                # 模拟模式：只存储在内存中
                self.mock_docs.append({
                    "id": doc_id,
                    "type": "image",
                    "path": image_path,
                    "filename": image_basename,
                    "description": description,
                    "embedding": image_embedding,
                    "cluster_id": cluster_id,
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
        """檢查新增圖片與現有圖片的相似度
        
        注意：在 Milvus 中，我們不預先建立關係，而是在查詢時動態計算相似度
        此方法保留用於兼容性，但實際上不會創建永久性的關係
        """
        if self.use_mock:
            return
        
        try:
            # 使用 Milvus 搜索相似圖片
            collection = Collection(self.collection_name)
            
            # 確保集合已加載
            try:
                collection.load()
            except Exception as load_error:
                print(f"加載集合時發生錯誤: {load_error}")
                # 嘗試創建索引
                try:
                    self._create_index(collection)
                except Exception as index_error:
                    print(f"創建索引時發生錯誤: {index_error}")
                    return
            
            try:
                # 設置搜索參數
                search_params = {
                    "metric_type": "COSINE",  # 使用餘弦相似度
                    "params": {"nprobe": 10}  # 搜索參數，可根據需求調整
                }
                
                # 執行向量搜索
                results = collection.search(
                    data=[image_embedding.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=20,
                    expr="type == 'image'"  # 只搜索圖片
                )
                
                # 提取相似度結果用於日誌記錄
                if results and len(results) > 0:
                    similar_results = []
                    for i, hits in enumerate(results):
                        for hit in hits:
                            if hit.score >= 0.75:  # 相似度閾值
                                similar_id = str(hit.id)
                                similarity = hit.score
                                similar_results.append((similar_id, similarity))
                                print(f"發現相似圖片: {image_id} ~ {similarity:.4f} ~ {similar_id}")
                    
                    print(f"找到 {len(similar_results)} 個相似圖片")
            except Exception as search_error:
                print(f"搜索相似圖片時發生錯誤: {search_error}")
            
            # 釋放集合
            collection.release()
        except Exception as e:
            print(f"使用 Milvus 計算相似度失敗: {e}")
    
    def add_video(self, video_path, metadata=None):
        """添加视频到向量数据库"""
        if metadata is None:
            metadata = {}
            
        # 编码视频
        video_embedding = self.encoder.encode_video(video_path)
        
        # 视频没有直接的文本描述，可以从元数据或文件名生成
        title = metadata.get("title", os.path.basename(video_path))
        description = metadata.get("description", f"Video file: {title}")
        
        # 生成唯一ID
        numeric_id = np.random.randint(10000, 9999999)
        doc_id = f"vid_{os.path.basename(video_path)}_{numeric_id}"
        
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
        
        if not self.use_mock:
            try:
                # 準備插入到 Milvus 的數據
                current_time = int(time.time())
                
                # 獲取向量維度，確保與 schema 一致
                embedding_list = video_embedding.tolist()
                
                # 創建符合 Milvus 要求的數據格式
                insert_data = [
                    [numeric_id],  # id 必須是 INT64
                    [embedding_list],  # embedding 是向量
                    [current_time],  # created_time 是 INT64
                    [current_time],  # updated_time 是 INT64
                    [str(description)],  # description 是 VARCHAR
                    [str(os.path.basename(video_path))],  # filename 是 VARCHAR
                    [str(video_path)],  # path 是 VARCHAR
                    ["video"]  # type 是 VARCHAR
                ]
                
                # 向 Milvus 插入數據
                try:
                    collection = Collection(self.collection_name)
                    
                    # 嘗試插入數據
                    try:
                        collection.insert(insert_data)
                        collection.flush()
                        print(f"影片 {title} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                    except Exception as insert_error:
                        print(f"插入數據時發生錯誤: {insert_error}")
                        # 嘗試重新建立索引
                        try:
                            self._create_index(collection)
                            # 再次嘗試插入
                            collection.insert(insert_data)
                            collection.flush()
                            print(f"重新建立索引後，影片 {title} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                        except Exception as retry_error:
                            print(f"重新嘗試插入時發生錯誤: {retry_error}")
                            raise retry_error
                    finally:
                        # 確保在所有情況下都釋放資源
                        try:
                            collection.release()
                        except Exception as release_error:
                            print(f"釋放集合時發生錯誤: {release_error}")
                except Exception as collection_error:
                    print(f"獲取集合時發生錯誤: {collection_error}")
                    # 嘗試重新創建集合
                    try:
                        self._create_collection()
                        # 獲取新創建的集合
                        collection = Collection(self.collection_name)
                        collection.insert(insert_data)
                        collection.flush()
                        collection.release()
                        print(f"重新創建集合後，影片 {title} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                    except Exception as recreate_error:
                        print(f"重新創建集合後插入數據時發生錯誤: {recreate_error}")
                        raise recreate_error
            except Exception as e:
                print(f"添加影片到 Milvus 時發生錯誤: {e}")
        else:
            # 模拟模式：只存储在内存中
            self.mock_docs.append({
                "id": doc_id,
                "type": "video",
                "path": video_path,
                "title": title,
                "description": description,
                "embedding": video_embedding,
                "metadata": metadata
            })
            print(f"影片 {title} 已成功添加到模擬數據庫，ID: {doc_id}")
        
        return doc_id
    
    def add_youtube_video(self, youtube_url, metadata=None):
        """添加YouTube视频到向量数据库"""
        # 下载并编码YouTube视频
        embedding, yt_metadata = self.encoder.encode_youtube_video(youtube_url)
        
        # 合并元数据
        if metadata is None:
            metadata = {}
        metadata = {**yt_metadata, **metadata}
        
        # 生成唯一ID
        video_id = youtube_url.split("v=")[-1] if "v=" in youtube_url else "unknown"
        numeric_id = np.random.randint(10000, 9999999)
        doc_id = f"yt_{video_id}_{numeric_id}"
        
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
        
        if not self.use_mock:
            try:
                # 準備插入到 Milvus 的數據
                current_time = int(time.time())
                
                # 標題與描述
                title = metadata.get("title", "YouTube Video")
                description = metadata.get("description", f"YouTube video: {title}")
                
                # 處理嵌入向量，確保格式正確
                if embedding is not None:
                    embedding_list = embedding.tolist()
                else:
                    # 如果沒有嵌入向量，創建一個空向量（需要符合集合的維度）
                    embedding_list = [0.0] * 512  # 假設向量維度為 512
                
                # 創建符合 Milvus 要求的數據格式
                insert_data = [
                    [numeric_id],  # id 必須是 INT64
                    [embedding_list],  # embedding 是向量
                    [current_time],  # created_time 是 INT64
                    [current_time],  # updated_time 是 INT64
                    [str(description)],  # description 是 VARCHAR
                    [f"{video_id}.mp4"],  # filename 是 VARCHAR
                    [str(youtube_url)],  # path 是 VARCHAR
                    ["youtube_video"]  # type 是 VARCHAR
                ]
                
                # 向 Milvus 插入數據
                try:
                    collection = Collection(self.collection_name)
                    
                    # 嘗試插入數據
                    try:
                        collection.insert(insert_data)
                        collection.flush()
                        print(f"YouTube 影片 {title} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                    except Exception as insert_error:
                        print(f"插入數據時發生錯誤: {insert_error}")
                        # 嘗試重新建立索引
                        try:
                            self._create_index(collection)
                            # 再次嘗試插入
                            collection.insert(insert_data)
                            collection.flush()
                            print(f"重新建立索引後，YouTube 影片 {title} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                        except Exception as retry_error:
                            print(f"重新嘗試插入時發生錯誤: {retry_error}")
                            raise retry_error
                    finally:
                        # 確保在所有情況下都釋放資源
                        try:
                            collection.release()
                        except Exception as release_error:
                            print(f"釋放集合時發生錯誤: {release_error}")
                except Exception as collection_error:
                    print(f"獲取集合時發生錯誤: {collection_error}")
                    # 嘗試重新創建集合
                    try:
                        self._create_collection()
                        # 獲取新創建的集合
                        collection = Collection(self.collection_name)
                        collection.insert(insert_data)
                        collection.flush()
                        collection.release()
                        print(f"重新創建集合後，YouTube 影片 {title} 已成功添加到 Milvus 數據庫，ID: {doc_id}")
                    except Exception as recreate_error:
                        print(f"重新創建集合後插入數據時發生錯誤: {recreate_error}")
                        raise recreate_error
            except Exception as e:
                print(f"添加 YouTube 影片到 Milvus 時發生錯誤: {e}")
        else:
            # 模拟模式
            self.mock_docs.append({
                "id": doc_id,
                "type": "youtube_video",
                "url": youtube_url,
                "embedding": embedding,
                "metadata": metadata
            })
            print(f"YouTube 影片 {metadata.get('title', 'unknown')} 已成功添加到模擬數據庫，ID: {doc_id}")
        
        return doc_id

    def get_all_images(self, limit=50):
        """獲取數據庫中所有的圖片"""
        if self.use_mock:
            # 返回模拟数据中所有类型为 "image" 的文档
            return [doc for doc in self.mock_docs if doc.metadata.get("type") == "image"][:limit]
        
        try:
            # 使用 Milvus 查詢所有圖片
            try:
                collection = Collection(self.collection_name)
                
                # 嘗試加載集合
                try:
                    collection.load()
                    
                    # 查詢所有類型為 'image' 的記錄
                    try:
                        results = collection.query(
                            expr="type == 'image'",
                            output_fields=["id", "path", "description", "filename", "embedding", "cluster_id", "created_time", "updated_time"],
                            limit=limit
                        )
                        
                        # 轉換為 Document 對象
                        docs = []
                        for record in results:
                            # 檢查路徑是否存在
                            path = record.get("path", "")
                            if path and os.path.exists(path):
                                # 創建元數據字典
                                metadata = {
                                    "doc_id": f"img_{record.get('id')}",
                                    "type": "image",
                                    "path": path,
                                    "filename": record.get("filename", ""),
                                    "cluster_id": record.get("cluster_id", 0),
                                    "created_time": record.get("created_time", 0),
                                    "updated_time": record.get("updated_time", 0),
                                    "embedding": record.get("embedding")
                                }
                                
                                # 建立 Document 對象
                                doc = Document(
                                    page_content=record.get("description", ""),
                                    metadata=metadata
                                )
                                docs.append(doc)
                        
                    except Exception as query_error:
                        print(f"查詢圖片時發生錯誤: {query_error}")
                        # 嘗試重新建立索引後再查詢
                        try:
                            self._create_index(collection)
                            collection.load()
                            
                            # 再次嘗試查詢
                            results = collection.query(
                                expr="type == 'image'",
                                output_fields=["id", "path", "description", "filename", "embedding", "cluster_id", "created_time", "updated_time"],
                                limit=limit
                            )
                            
                            # 轉換為 Document 對象
                            docs = []
                            for record in results:
                                # 檢查路徑是否存在
                                path = record.get("path", "")
                                if path and os.path.exists(path):
                                    # 創建元數據字典
                                    metadata = {
                                        "doc_id": f"img_{record.get('id')}",
                                        "type": "image",
                                        "path": path,
                                        "filename": record.get("filename", ""),
                                        "cluster_id": record.get("cluster_id", 0),
                                        "created_time": record.get("created_time", 0),
                                        "updated_time": record.get("updated_time", 0),
                                        "embedding": record.get("embedding")
                                    }
                                    
                                    # 建立 Document 對象
                                    doc = Document(
                                        page_content=record.get("description", ""),
                                        metadata=metadata
                                    )
                                    docs.append(doc)
                        except Exception as retry_error:
                            print(f"重建索引後查詢時發生錯誤: {retry_error}")
                            docs = []
                
                except Exception as load_error:
                    print(f"加載集合時發生錯誤: {load_error}")
                    docs = []
                
                finally:
                    # 確保釋放資源
                    try:
                        collection.release()
                    except Exception as release_error:
                        print(f"釋放集合時發生錯誤: {release_error}")
                
                return docs
                
            except Exception as collection_error:
                print(f"獲取集合時發生錯誤: {collection_error}")
                # 嘗試重新創建集合
                try:
                    self._create_collection()
                    return []  # 新創建的集合中沒有圖片
                except Exception as recreate_error:
                    print(f"重新創建集合時發生錯誤: {recreate_error}")
                    return []
                
        except Exception as e:
            print(f"獲取所有圖片時發生錯誤: {e}")
            return []
            
    def get_all_videos(self, limit=50):
        """獲取數據庫中所有的影片"""
        if self.use_mock:
            # 返回模拟数据中所有类型为 "video" 的文档
            return [doc for doc in self.mock_docs if doc.metadata.get("type") == "video"][:limit]
        
        try:
            # 使用 Milvus 查詢所有影片
            try:
                collection = Collection(self.collection_name)
                
                # 嘗試加載集合
                try:
                    collection.load()
                    
                    # 查詢所有類型為 'video' 的記錄
                    try:
                        results = collection.query(
                            expr="type == 'video'",
                            output_fields=["id", "path", "description", "filename", "embedding", "created_time", "updated_time"],
                            limit=limit
                        )
                        
                        # 轉換為 Document 對象
                        docs = []
                        for record in results:
                            # 檢查路徑是否存在
                            path = record.get("path", "")
                            if path and os.path.exists(path):
                                # 從路徑取得標題
                                title = os.path.basename(path)
                                
                                # 創建元數據字典
                                metadata = {
                                    "doc_id": f"vid_{record.get('id')}",
                                    "type": "video",
                                    "path": path,
                                    "title": title,
                                    "filename": record.get("filename", ""),
                                    "created_time": record.get("created_time", 0),
                                    "updated_time": record.get("updated_time", 0)
                                }
                                
                                # 建立 Document 對象
                                doc = Document(
                                    page_content=record.get("description", ""),
                                    metadata=metadata
                                )
                                docs.append(doc)
                        
                    except Exception as query_error:
                        print(f"查詢影片時發生錯誤: {query_error}")
                        # 嘗試重新建立索引後再查詢
                        try:
                            self._create_index(collection)
                            collection.load()
                            
                            # 再次嘗試查詢
                            results = collection.query(
                                expr="type == 'video'",
                                output_fields=["id", "path", "description", "filename", "embedding", "created_time", "updated_time"],
                                limit=limit
                            )
                            
                            # 轉換為 Document 對象
                            docs = []
                            for record in results:
                                # 檢查路徑是否存在
                                path = record.get("path", "")
                                if path and os.path.exists(path):
                                    # 從路徑取得標題
                                    title = os.path.basename(path)
                                    
                                    # 創建元數據字典
                                    metadata = {
                                        "doc_id": f"vid_{record.get('id')}",
                                        "type": "video",
                                        "path": path,
                                        "title": title,
                                        "filename": record.get("filename", ""),
                                        "created_time": record.get("created_time", 0),
                                        "updated_time": record.get("updated_time", 0)
                                    }
                                    
                                    # 建立 Document 對象
                                    doc = Document(
                                        page_content=record.get("description", ""),
                                        metadata=metadata
                                    )
                                    docs.append(doc)
                        except Exception as retry_error:
                            print(f"重建索引後查詢時發生錯誤: {retry_error}")
                            docs = []
                
                except Exception as load_error:
                    print(f"加載集合時發生錯誤: {load_error}")
                    docs = []
                
                finally:
                    # 確保釋放資源
                    try:
                        collection.release()
                    except Exception as release_error:
                        print(f"釋放集合時發生錯誤: {release_error}")
                
                return docs
                
            except Exception as collection_error:
                print(f"獲取集合時發生錯誤: {collection_error}")
                # 嘗試重新創建集合
                try:
                    self._create_collection()
                    return []  # 新創建的集合中沒有影片
                except Exception as recreate_error:
                    print(f"重新創建集合時發生錯誤: {recreate_error}")
                    return []
                
        except Exception as e:
            print(f"獲取所有影片時發生錯誤: {e}")
            return []
    
    def get_cluster_statistics(self):
        """獲取聚類統計信息"""
        if self.use_mock:
            return {"message": "模擬模式下無聚類統計"}
        
        try:
            cluster_collection = Collection(self.cluster_collection_name)
            cluster_collection.load()
            
            # 查詢所有聚類
            results = cluster_collection.query(
                expr="cluster_id >= 0",
                output_fields=["cluster_id", "count", "created_time", "updated_time"],
                limit=1000
            )
            
            cluster_collection.release()
            
            if results:
                total_clusters = len(results)
                total_images_in_clusters = sum(record.get("count", 0) for record in results)
                avg_cluster_size = total_images_in_clusters / total_clusters if total_clusters > 0 else 0
                
                return {
                    "total_clusters": total_clusters,
                    "total_images_in_clusters": total_images_in_clusters,
                    "average_cluster_size": round(avg_cluster_size, 2),
                    "clusters": results
                }
            else:
                return {
                    "total_clusters": 0,
                    "total_images_in_clusters": 0,
                    "average_cluster_size": 0,
                    "clusters": []
                }
                
        except Exception as e:
            self.logger.error(f"獲取聚類統計時發生錯誤: {e}")
            return {"error": str(e)}
    
    def get_images_by_cluster(self, cluster_id, limit=50):
        """獲取指定聚類中的所有圖片"""
        if self.use_mock:
            return [doc for doc in self.mock_docs 
                   if doc.get("cluster_id") == cluster_id and doc.get("type") == "image"][:limit]
        
        try:
            collection = Collection(self.collection_name)
            collection.load()
            
            # 查詢指定聚類中的圖片
            results = collection.query(
                expr=f"cluster_id == {cluster_id} and type == 'image'",
                output_fields=["id", "path", "description", "filename", "embedding", "cluster_id", "created_time", "updated_time"],
                limit=limit
            )
            
            # 轉換為 Document 對象
            docs = []
            for record in results:
                path = record.get("path", "")
                if path and os.path.exists(path):
                    metadata = {
                        "doc_id": f"img_{record.get('id')}",
                        "type": "image",
                        "path": path,
                        "filename": record.get("filename", ""),
                        "cluster_id": record.get("cluster_id", 0),
                        "created_time": record.get("created_time", 0),
                        "updated_time": record.get("updated_time", 0),
                        "embedding": record.get("embedding")
                    }
                    
                    doc = Document(
                        page_content=record.get("description", ""),
                        metadata=metadata
                    )
                    docs.append(doc)
            
            collection.release()
            return docs
            
        except Exception as e:
            self.logger.error(f"獲取聚類 {cluster_id} 中的圖片時發生錯誤: {e}")
            return []


    def delete_image(self, doc_id):
        """
        從 Milvus 刪除圖片
        
        Args:
            doc_id: 圖片的文檔ID
            
        Returns:
            bool: 是否成功刪除
        """
        if self.use_mock:
            # 模拟模式下从内存中删除
            for i, doc in enumerate(self.mock_docs):
                if doc.get("id") == doc_id:
                    # 獲取圖片所屬的聚類ID
                    cluster_id = doc.get("cluster_id", 0)
                    # 更新模擬聚類計數
                    if cluster_id > 0:
                        self.logger.info(f"在模擬模式下減少聚類 {cluster_id} 的計數")
                    del self.mock_docs[i]
                    return True
            return False
        
        try:
            # 從doc_id解析出Milvus的數字ID
            # 假設doc_id格式為 "img_123456"
            if not doc_id.startswith("img_"):
                return False
                
            numeric_id = doc_id[4:]  # 去掉"img_"前綴
            
            try:
                numeric_id = int(numeric_id)
            except ValueError:
                # 如果ID包含其他信息（如img_file_123456），嘗試提取數字部分
                import re
                match = re.search(r'(\d+)', numeric_id)
                if match:
                    numeric_id = int(match.group(1))
                else:
                    self.logger.error(f"無法從doc_id '{doc_id}' 解析出數字ID")
                    return False
            
            # 首先獲取圖片所屬的聚類ID
            collection = Collection(self.collection_name)
            collection.load()
            
            # 查詢圖片資訊
            results = collection.query(
                expr=f"id == {numeric_id}",
                output_fields=["cluster_id"]
            )
            
            cluster_id = 0
            if results and len(results) > 0:
                cluster_id = results[0].get("cluster_id", 0)
            
            # 刪除圖片記錄
            expr = f"id == {numeric_id}"
            collection.delete(expr)
            collection.flush()  # 確保刪除操作生效
            collection.release()
            
            # 如果圖片屬於某個聚類，更新聚類計數
            if cluster_id > 0:
                self._update_cluster_count_after_delete(cluster_id)
            
            self.logger.info(f"已從Milvus刪除ID為 {numeric_id} 的圖片，cluster_id: {cluster_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"刪除圖片時發生錯誤: {e}")
            return False
    
    def _update_cluster_count_after_delete(self, cluster_id):
        """
        在刪除圖片後更新聚類的計數
        
        Args:
            cluster_id: 聚類ID
        """
        try:
            cluster_collection = Collection(self.cluster_collection_name)
            cluster_collection.load()
            
            # 獲取當前聚類信息
            results = cluster_collection.query(
                expr=f"cluster_id == {cluster_id}",
                output_fields=["cluster_id", "centroid_vector", "count", "created_time", "updated_time"]
            )
            
            if results and len(results) > 0:
                current_count = results[0].get("count", 0)
                centroid_vector = results[0].get("centroid_vector")
                created_time = results[0].get("created_time", 0)
                
                if current_count <= 1:
                    # 如果這是聚類中的最後一張圖片，刪除整個聚類
                    cluster_collection.delete(f"cluster_id == {cluster_id}")
                    self.logger.info(f"刪除了空聚類 {cluster_id}")
                else:
                    # 否則減少計數
                    new_count = current_count - 1
                    current_time = int(time.time())
                    
                    # 刪除舊記錄
                    cluster_collection.delete(f"cluster_id == {cluster_id}")
                    cluster_collection.flush()
                    
                    # 插入更新後的記錄
                    insert_data = [
                        [cluster_id],                    # cluster_id
                        [centroid_vector],               # centroid_vector (保持不變)
                        [new_count],                     # count (減1)
                        [created_time],                  # created_time (保持不變)
                        [current_time]                   # updated_time (更新)
                    ]
                    
                    cluster_collection.insert(insert_data)
                    cluster_collection.flush()
                    
                    self.logger.info(f"更新聚類 {cluster_id} 計數: {current_count} -> {new_count}")
            
            cluster_collection.release()
            
        except Exception as e:
            self.logger.error(f"更新聚類計數時發生錯誤: {e}")

    def __del__(self):
        """清理連接"""