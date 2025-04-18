from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-7dfe20a081554d3e896c7044ed951b0a",
    model="text-embedding-v3")

# 简单的基础链
first_prompt = ChatPromptTemplate.from_messages([
  ("human", "帮我给{type}生产的公司,起一个容易记忆的名字")
])
first_llm_chain = LLMChain(llm=llm,
                           prompt=first_prompt,
                           # 是否开启日志
                           verbose=True)

second_prompt = ChatPromptTemplate.from_messages([
  ("human", "用5个词来貌似下这个公司名字:{company_name}")
])
second_llm_chain = LLMChain(llm=llm,
                           prompt=second_prompt,
                           # 是否开启日志
                           verbose=True)

overall_simple_chain = SimpleSequentialChain(
    chains=[first_llm_chain, second_llm_chain],
    verbose=True
)
print(overall_simple_chain.run("肥皂"))
