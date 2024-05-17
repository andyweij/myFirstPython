import gdown
import zipfile
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import csv
from sentence_transformers import SentenceTransformer
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from env_settings import EnvSettings

# url = 'https://drive.google.com/uc?id=11ISS45aO2ubNCGaC3Lvd3D7NT8Y7MeO8'
# output = './movies.zip'
# gdown.download(url, output)
#
# with zipfile.ZipFile("./movies.zip", "r") as zip_ref:
#     zip_ref.extractall("./movies")

# Milvus Setup Arguments
COLLECTION_NAME = 'movies_db'  # Collection name
DIMENSION = 384  # Embeddings size
COUNT = 1000  # Number of vectors to insert
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

# Inference Arguments
BATCH_SIZE = 128

# Search Arguments
TOP_K = 5

# Connect to Milvus Database
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Remove any previous collections with the same name
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=200),
    # VARCHARS need a maximum length, so for this example they are set to 200 characters
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name='plot', dtype=DataType.VARCHAR, max_length=10000),
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
# Create an IVF_FLAT index for collection.
index_params = {
    'metric_type': 'L2',
    'index_type': "IVF_FLAT",
    'params': {'nlist': 1536}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

transformer = SentenceTransformer('all-MiniLM-L6-v2')


# Extract the book titles
def csv_load(file):
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if '' in (row[1], row[7]):
                continue
            yield row[1], row[7]


# Extract embedding from text using OpenAI
def embed_insert(data):
    embeds = transformer.encode(data[1])
    ins = [
        data[0],
        [x for x in embeds],
        data[1]
    ]
    collection.insert(ins)


data_batch = [[], []]

count = 0

for title, plot in csv_load('./movies/plots.csv'):
    if count <= COUNT:
        data_batch[0].append(title)
        data_batch[1].append(plot)
        if len(data_batch[0]) % BATCH_SIZE == 0:
            embed_insert(data_batch)
            data_batch = [[], [], []]
        count += 1
    else:
        break

# Embed and insert the remainder
if len(data_batch[0]) != 0:
    embed_insert(data_batch)

# Call a flush to index any unsealed segments.
collection.flush()

# Search for titles that closest match these phrases.
search_terms = ['a man']


# Search the database based on input text
def embed_search(data):
    embeds = transformer.encode(data)
    return [x for x in embeds]


search_data = embed_search(search_terms)

start = time.time()
res = collection.search(
    data=search_data,  # Embeded search value
    anns_field="embedding",  # Search across embeddings
    param={},
    limit=TOP_K,  # Limit to top_k results per search
    output_fields=['title', 'plot']  # Include title field in result
)
end = time.time()
askContent = ''
for hits_i, hits in enumerate(res):
    print('Title:', search_terms[hits_i])
    print('Search Time:', end - start)
    print('Results:')
    count = 1
    for hit in hits:
        askContent += str(count) + ':' + hit.entity.get('title') + '\n'
        count += 1
        # print(hit.entity.get('title'), '----', hit.distance)
        # print(hit.entity.get('plot'), '----', hit.distance)

#
print('------------------\n')
prompt = '你知道以下電影嗎 \n'
print(prompt+askContent)
env_settings = EnvSettings()
llm = ChatGoogleGenerativeAI(google_api_key=env_settings.GOOGLE_API_KEY,model="gemini-pro")
result = llm.invoke(prompt+askContent)
print(result.content)
#
print('------------------\n')
# for chunk in llm.stream("請幫我使用中文解釋，langchain裡的Streaming and Batching這項功能是做什麼的"):
#     print(chunk.content)
