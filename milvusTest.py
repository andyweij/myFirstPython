from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection


embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

search_terms = '垃圾車'
# vector_db = Milvus(
#     embeddings,
#     connection_args={"host": "localhost", "port": "19530"},
#     collection_name="collection_1",
# )
# connection_args = {
#             "host": 'localhost',
#             "port": '19530'
#         }
# connections.connect(host='localhost', port='19530')
# fields = [
#     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
# ]
# schema = CollectionSchema(fields, description="test collection")
#
# target_collection = Collection(name='test', schema=schema)
#
# index_params = {
#     "metric_type": "L2",
#     "index_type": "HNSW",
#     "params": {"M": 8, "efConstruction": 64},
#     "index_name": "_default_idx_104"
# }
# target_collection.create_index(field_name="vector", index_params=index_params)
# target_collection.load()
# chunk_content='這段程式碼的主要功能是將長文本 chunk_content 分割成小塊，然後使用 Milvus 將每個小塊的嵌入向量存儲到指定的集合中。這樣可以實現高效的文本搜尋和相似度匹配功能。'
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0)
# documents = text_splitter.create_documents(
#     texts=[chunk_content],
#     metadatas=[]
# )
#
# vector_store = Milvus(
#     embedding_function=embeddings,
#     collection_name='test',
#     connection_args=connection_args,
#     auto_id=True
# )
# vector_store.add_documents(documents)

vector_db = Milvus(
    embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name='tainanDB',
    primary_field="id",
    vector_field='embedding',
    text_field='subject'
)

docs = vector_db.similarity_search(query=search_terms, k=5)

for row in docs:
    print(row.page_content+':')

