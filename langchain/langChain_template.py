import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, \
  MessagesPlaceholder
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

# llm = Tongyi(api_key="sk-7dfe20a081554d3e896c7044ed951b0a")

# my_prompt = PromptTemplate.from_template("翻译下面这段话为中文:{input}")

my_prompt = ChatPromptTemplate.from_messages([
  ("system", "你是我的翻译官,如果是中文翻译成英文,如果是英文,翻译成中文"),
  ("user", "{input}"),
  # 可传如一组消息
  # MessagesPlaceholder("array")
])

#  StrOutputParser 帮忙做格式化,没有多余的
chain = my_prompt | llm | StrOutputParser()

result = chain.invoke({"input": "Hello world!"})
print(result)