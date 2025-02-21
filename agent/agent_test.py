import os

from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

os.environ["SERPAPI_API_KEY"] = "f1fd465147cd94025287abbe521d75c65e63471dd497c0c9903d5034d3e9e7ba"
llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    # 工具包
    tools,
    # 模型
    llm,
    # 多类型
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # 开启日志
    verbose=True,
)
result = agent.run("现任的美国总统是谁,年纪多大？")
print(result)