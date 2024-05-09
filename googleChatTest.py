# %%
import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from IPython.display import Image
from env_settings import EnvSettings
# if "GOOGLE_API_KEY" not in os.environ:
# os.environ["GOOGLE_API_KEY"] = getpass.getpass()
env_settings = EnvSettings()
print('第一段:\n')
llm = ChatGoogleGenerativeAI(google_api_key=env_settings.GOOGLE_API_KEY,model="gemini-pro")
result = llm.invoke("請用中文幫我介紹 LangChain")
print(result.content)

print('第二段:\n')
for chunk in llm.stream("請幫我使用中文解釋，langchain裡的Streaming and Batching這項功能是做什麼的"):
    print(chunk.content)
    print("---第三段")

results = llm.batch(
    [
        "What's 2+2?",
        "What's 3+5?",
    ]
)
for res in results:
    print(res.content)

# image_url = "https://picsum.photos/seed/picsum/300/300"
# content = requests.get(image_url).content
# Image(content)
#
# llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", api_key="AIzaSyB1TfkXS_LtbY1VsAgZQtAH3bGng0vzusY")
# # example
# message = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": "What's in this image?",
#         },  # You can optionally provide text parts
#         {"type": "image_url", "image_url": image_url},
#     ]
# )
# llm.invoke([message])