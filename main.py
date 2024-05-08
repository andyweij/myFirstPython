#%%
import getpass
import os
import google.generativeai as genai
import test
import googleChatTest

# if __name__ == '__main__':

content = input("Enter you Question:")
response = test.generate_content(content)
print("\n")
print(response)
