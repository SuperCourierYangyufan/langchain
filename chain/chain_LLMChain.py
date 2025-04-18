from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import  ChatPromptTemplate
from pydantic import SecretStr

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-7dfe20a081554d3e896c7044ed951b0a",
    model="text-embedding-v3")


prompt = ChatPromptTemplate.from_messages([
  ("human","将一个短小精悍关于{type}的故事")
])

llm_chain = LLMChain(
    llm = llm,
    prompt=prompt,
    # 是否开启日志
    verbose=True
)
result = llm_chain.invoke({"type":"未来"})
print(result)
