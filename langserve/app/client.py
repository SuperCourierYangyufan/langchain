from gc import set_debug

import requests
from langchain.agents import create_tool_calling_agent
from langchain.globals import set_verbose
from langchain.schema.runnable import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve.client import RemoteRunnable
#----------------------python内调用---------------------------
# 本地地址+server.py中router的path地址
# openai = RemoteRunnable("http://localhost:8000/openai/")
#
# prompt = ChatPromptTemplate.from_messages(
#   [
#     ("system", "你是一个故事助手"),
#     ("user", "写一个关于{topic}的故事")
#   ]
# )
# chain = prompt | RunnableMap({"openai": openai})
# response = chain.invoke({"topic": "猫"})
# print(response)
# #----------------------rest内调用普通---------------------------
#  本地地址+server.py中router的path地址+invoke
# response = requests.post("http://localhost:8000/openai_ext/invoke",
#               json={"input":{"topic":"猫"}})
# print(response.json())
# #----------------------流式---------------------------
# openai = RemoteRunnable("http://127.0.0.1:8000/openai_ext/")|StrOutputParser()
# chain =  RunnableMap({"openai": openai})
# for chunk in chain.stream({"topic": "猫"}):
#     print(chunk)