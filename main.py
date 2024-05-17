#%%
import getpass
import os
import google.generativeai as genai
import test
import googleChatTest
from openai import AzureOpenAI

# if __name__ == '__main__':

# content = input("Enter you Question:")
# response = test.generate_content(content)
# print("\n")
# print(response)
# client = AzureOpenAI(
#     api_key="8a9ecacc49ab4ca290933af0ad820e53",
#     api_version="2024-02-01",
#     azure_endpoint="https://tainan-openai-001.openai.azure.com/"
# )
#
# response = client.chat.completions.create(
#     model="tainan-gpt-35-16k",  # model = "deployment_name".
#     messages=[
#         {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
#         {"role": "user", "content": "Who were the founders of Microsoft?"}
#     ]
# )

# print(response)
# print(response.model_dump_json(indent=2))
# print(response.choices[0].message.content)
