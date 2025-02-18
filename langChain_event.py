import asyncio

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))


# 输出所有事件
async def async_stream():
  events = []
  async for event in llm.astream_events("hello", version="v2"):
    events.append(event)
  print(events)

asyncio.run(async_stream())