import os
from dotenv import load_dotenv
from google import genai

def doTheAiStuff(prompt:str): 
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    # response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def doTheAiStuff2(prompt): 
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    # response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response