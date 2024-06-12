# %%
# import getpass
# import os
# import google.generativeai as genai
# import test
# import googleChatTest
# from openai import AzureOpenAI
import test
from test import Point

if __name__ == '__main__':
    list1 = [1, 3, 5, 7]
    print(list1[::1])
    a: bool = True
    result = (1, 2)[a]
    print(result)
    p1 = Point(0, 0)
    p2 = Point(0, 0)
    p3 = Point(x=0, y=5)
    p4 = Point(y=5, x=0)
    test.where_is(p3)
    print(p1)
    print(p2)
    print(p3)
    print(p4)

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
