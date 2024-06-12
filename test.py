# import os
# import google.generativeai as genai
# from env_settings import EnvSettings
# import os


#
# def generate_content(content):
#     prompt = ("1.請使用中文回答"
#               "2.請舉範例\n")
#     env_settings = EnvSettings()
#     genai.configure(api_key=env_settings.GOOGLE_API_KEY, transport='rest')
#     model = genai.GenerativeModel('gemini-pro')
#     response = model.generate_content(prompt + content)
#     # print(response.text)
#     return response.text
#
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def where_is(point):
    match point:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=0, y=y):
            print(f"Y={y}")
        case Point(x=x, y=0):
            print(f"X={x}")
        case Point():
            print("Somewhere else")
        case _:
            print("Not a point")
