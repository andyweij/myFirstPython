from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from env_settings import EnvSettings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pymysql

db_settings = {
    "host": "10.20.80.22",
    "port": 3306,
    "user": "sv",
    "password": "eland4321",
    "db": "tnc_1999_data",
    "charset": "utf8mb4",
}

# 建立Connection物件
conn = pymysql.connect(**db_settings)
# 建立Cursor物件
with conn.cursor() as cursor:
    # 查詢資料SQL語法
    command = "SELECT sid , subject , subItemName,retuData FROM tnc_1999_data.content2 where createTime >'2024-05-13'"
    # 執行指令
    cursor.execute(command)
    # 取得所有資料
    result = cursor.fetchall()
    # print(result)

env_settings = EnvSettings()
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

connection_args = {
    "host": 'localhost',
    "port": '19530'
}
connections.connect(host='localhost', port='19530')

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='sid', dtype=DataType.INT64),
    FieldSchema(name='subject', dtype=DataType.VARCHAR, max_length=50000),
    FieldSchema(name='subItemName', dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name='retuData', dtype=DataType.VARCHAR, max_length=50000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=384),
]
schema = CollectionSchema(fields, description="test collection")

target_collection = Collection(name='test', schema=schema)

index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
    "index_name": "_default_idx_104"
}
target_collection.create_index(field_name="embedding", index_params=index_params)
target_collection.load()

text_splitter = RecursiveCharacterTextSplitter()
all_documents = []
for row in result:
    sid, subject, subItemName, retuData = row
    text = f"Subject: {subject}"
    metadata = {"RetuData": retuData, "subItemName": subItemName, "sid": sid}

    documents = text_splitter.create_documents(
        texts=[subject],
        metadatas=[dict(file_chunk_id=sid, file_id=sid, sid=sid, subject=subject,
                        subItemName=subItemName,
                        retuData=retuData, )],
    )

    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name='test',
        connection_args=connection_args,
        auto_id=True,
        primary_field="id",
        text_field='subject',
        vector_field="embedding"
    )
    vector_store.add_documents(documents)
# bool_expr = "itemName like '%垃圾清運%'"
# search_terms = '垃圾亂丟'
# vector_db = Milvus(
#     embeddings,
#     connection_args={"host": "localhost", "port": "19530"},
#     collection_name='tainanDB',
#     primary_field="id",
#     vector_field='embedding',
#     text_field='subject'
# )
#
# docs = vector_db.similarity_search(query=search_terms, k=10, expr=bool_expr)
# retriever = vector_db.as_retriever()
#
# for row in docs:
#     print(row.page_content + ':')
#     print(row.metadata.get('subItemName'))
#
# print("----------------------")
# template = """Answer the question based only on the following context:
#
# 查詢後上下文:{context}
#
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
#
# llm = ChatGoogleGenerativeAI(google_api_key=env_settings.GOOGLE_API_KEY, model="gemini-pro")
#
#
# # 格式化文档
# def format_docs(docs):
#     formatted_docs = []
#     for doc in docs:
#         content = doc.page_content
#         metadata = doc.metadata
#         formatted_doc = f"問題: {content}\n回覆: {metadata.get('retuData', '')}"
#         formatted_docs.append(formatted_doc)
#     formatted_content = "\n\n".join(formatted_docs)
#     # print("Formatted Docs:\n", formatted_content)  # 打印格式化后
#     return formatted_content
#
#
# chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
# )
# print("------")
# print("invoke:" + chain.invoke("請幫我透過回覆的內容整理出回覆"))
