import os

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


os.environ["SERPAPI_API_KEY"] = "f1fd465147cd94025287abbe521d75c65e63471dd497c0c9903d5034d3e9e7ba"
llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))
# 记忆组件
memory = ConversationBufferMemory(
    memory_key="my_history",
    return_messages=True
)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    # 工具包
    tools,
    # 模型
    llm,
    # 多类型
    agent=AgentType.OPENAI_FUNCTIONS,
    # 记忆组件
    memory = memory,
    # 开启日志
    verbose=True,
    # 额外提示词key
    agent_kwargs= {
      "extra_prompt_messages": [
        MessagesPlaceholder(variable_name="my_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
      ]
    }
)
print(agent.agent.prompt.messages)