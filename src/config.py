import os
import configparser

# 取得專案根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 組合 config.ini 的路徑
config_path = os.path.join(BASE_DIR, 'config', 'config.ini')

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# 取得 Neo4j 設定
NEO4J_URI = config.get('neo4j', 'uri')
NEO4J_USERNAME = config.get('neo4j', 'username')
NEO4J_PASSWORD = config.get('neo4j', 'password')
NEO4J_DATABASE = config.get('neo4j', 'database')

# 取得模型設定
USE_LOCAL_LLM = config.getboolean('model', 'use_local_llm', fallback=True)
LOCAL_MODEL_PATH = config.get('model', 'local_model_path')
EMBEDDING_MODEL = config.get('model', 'embedding_model')

# 取得 HuggingFace Token
HUGGINGFACE_API_TOKEN = config.get('huggingface', 'api_token', fallback=None)
if HUGGINGFACE_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN