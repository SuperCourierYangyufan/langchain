from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))
# 记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history"
)
tools = load_tools([], llm=llm)
agent = initialize_agent(
    # 工具包
    tools,
    # 模型
    llm,
    # 多类型
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    # 记忆组件
    memory = memory,
    # 开启日志
    verbose=True,
)
result = agent.run("你好 你现在叫Tom")
print(result)
result = agent.run("你名字是什么")
print(result)