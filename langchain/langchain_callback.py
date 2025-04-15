from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

with get_openai_callback() as callback:
    response = llm.invoke("讲一笑话")
    print(response)
    print(callback)