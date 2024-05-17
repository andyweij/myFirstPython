import os
import google.generativeai as genai
from env_settings import EnvSettings
import os



def generate_content(content):
    prompt = ("1.請使用中文回答"
              "2.請舉範例\n")
    env_settings = EnvSettings()
    genai.configure(api_key=env_settings.GOOGLE_API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt + content)
    # print(response.text)
    return response.text

