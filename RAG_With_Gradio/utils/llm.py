import os
import time
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from langchain_together import Together
from langchain_openai import ChatOpenAI
load_dotenv()

class LLM:
    def __init__(self):
        self.local_model_path = os.getenv("MODEL_PATH")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def local(self):
        start = time.time()
        llm = CTransformers(
            model=self.local_model_path,
            config={
                "max_new_tokens": 4096,
                "temperature": 0.0,
                "context_length": 4096,
            },
        )
        print("Load time:", time.time() - start)
        return llm
#for TogetherAPI
    def together(self):
        return Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            together_api_key=self.api_key,
        )

    def openai(self):
        return ChatOpenAI(
            model="gpt-4o-mini",  # fast + cheap
            temperature=0,
            api_key=self.api_key
        )