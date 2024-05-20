# %%
import pymysql
from pymilvus import connections, db, utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from env_settings import EnvSettings

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
    command = "SELECT sid , subject , subItemName,retuData FROM tnc_1999_data.content2 where createTime >'2024-03-01'"
    # 執行指令
    cursor.execute(command)
    # 取得所有資料
    result = cursor.fetchall()
    # print(result)

# Milvus Setup Arguments

DIMENSION = 384  # Embeddings size
COUNT = 1000  # Number of vectors to insert

# Inference Arguments
BATCH_SIZE = 128
connections.connect(host='localhost', port='19530')
# Search Arguments
TOP_K = 10

existing_databases = db.list_database()
# Connect to Milvus Database
if "tainanDB" not in existing_databases:
    db.create_database("tainanDB")
db.using_database("tainanDB")
# print(db.list_database())


fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='sid', dtype=DataType.INT64),
    FieldSchema(name='subject', dtype=DataType.VARCHAR, max_length=50000),
    FieldSchema(name='subItemName', dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name='retuData', dtype=DataType.VARCHAR, max_length=50000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),

]
schema = CollectionSchema(fields=fields)
# if utility.has_collection("tainanDB"):
#     utility.drop_collection("tainanDB")

users = utility.list_usernames()
database=db.list_database()

if not utility.has_collection("tainanDB"):
    collection = Collection(name='tainanDB', schema=schema)
else:
    collection = Collection("tainanDB")

# # Create an IVF_FLAT index for collection.
index_params = {
    'metric_type': 'L2',
    'index_type': "IVF_FLAT",
    'params': {'nlist': 1536}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def embed_insert(data):
    embeds = transformer.encode(data[1])
    ins = [
        data[0],
        data[1],
        data[2],
        data[3],
        embeds,
    ]
    collection.insert(ins)


data_batch = [[], [], [], []]

count = 0

if collection.num_entities == 0:
    for sid, subject, subItemName, retuData in result:
        if count < COUNT:
            data_batch[0].append(sid)
            data_batch[1].append(subject)
            data_batch[2].append(subItemName)
            data_batch[3].append(retuData)
            if len(data_batch) % BATCH_SIZE == 0:
                embed_insert(data_batch)
                data_batch = [[], [], [], []]
            count += 1
        else:
            break
    # 插入剩余数据
    if len(data_batch[0]) != 0:
        embed_insert(data_batch)

    collection.flush()

search_terms = ['垃圾車']


def embed_search(data):
    embeds = transformer.encode(data)
    return [x for x in embeds]


search_data = embed_search(search_terms)

start = time.time()
bool_expr = "subItemName like '%垃圾清運%'"
res = collection.search(
    data=search_data,  # Embeded search value
    anns_field="embedding",  # Search across embeddings
    param={"metric_type": "L2"},
    limit=TOP_K,  # Limit to top_k results per search
    expr=bool_expr,
    output_fields=['id', 'sid', 'subject', 'subItemName', 'retuData']  # Include title field in result

)

end = time.time()

askContent = ''
for hits_i, hits in enumerate(res):
    print('subject:', search_terms[hits_i])
    print('Search Time:', end - start)
    print('Results:')
    count = 1
    for hit in hits:
        askContent += (str(hit.entity.get('sid')) + ':' + hit.entity.get('subject') + '\n' +
                       "subItemName:" + hit.entity.get(
                    'subItemName') + '\n' +
                       'ans:' + hit.entity.get('retuData') + '\n')
        count += 1

print(askContent)
print('------------------\n')
prompt = '請幫從提供的內容中，摘要出問題點列出\n'

env_settings = EnvSettings()
llm = ChatGoogleGenerativeAI(google_api_key=env_settings.GOOGLE_API_KEY, model="gemini-pro")
result = llm.invoke(prompt + askContent)
print(result.content)
