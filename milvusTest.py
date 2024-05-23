from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

search_terms = ['垃圾車']
vector_db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="collection_1",
)


vector_db = Milvus.from_documents(
    search_terms,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

docs = vector_db.similarity_search(query)

print(vector_db.alias('testalias'))
