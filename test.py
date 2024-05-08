import os
import google.generativeai as genai


def generate_content(content):
    prompt = ("1.請使用中文回答"
              "2.請舉範例\n")

    genai.configure(api_key='AIzaSyB1TfkXS_LtbY1VsAgZQtAH3bGng0vzusY', transport='rest')
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt+content)
    # print(response.text)
    return response.text
