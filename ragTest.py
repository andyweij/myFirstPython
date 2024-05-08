import bs4
from chromadb import Embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_google_vertexai import ChatVertexAI
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
# os.environ["GOOGLE_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
from sentence_transformers import SentenceTransformer, util
from PIL import Image

llm = ChatVertexAI(model="gemini-pro", project="my-project-rag-test")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
# for index, content in enumerate(splits):
#     print(index, ':', content)

# model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
model = SentenceTransformer('clip-ViT-B-32')

img_emb = model.encode(Image.open('Pug-dog.jpg'))
text_emb = model.encode(['Pug', 'house dog', 'small dog','a dog on the GROUND','Pug dog'])

#Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)
# Sentences we want to encode. Example:
# s1 = ['This framework generates embeddings for each input sentence']
# s2 = ['']
# # Sentences are encoded by calling model.encode()
# sentence1 = model.encode(s1)
# sentence2 = model.encode(s2)
# print(embedding.shape)
# cos_scores = util.cos_sim(img_emb, text_emb)
