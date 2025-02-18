import os

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.globals import set_debug, set_verbose
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

os.environ["TAVILY_API_KEY"] = "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"

openai = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))
tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages([
  ("system", "你是一位得力助手"),
  ("human", "{input}"),
  ("assistant", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
  openai,
  tools,
  prompt
)
set_debug(True)
set_verbose(False)
agent_executor = AgentExecutor(agent=agent, tools=tools)
response = agent_executor.invoke({"input": "写一个关于猫的故事"})
print(response)
