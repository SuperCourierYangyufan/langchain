import os

from langchain.agents import initialize_agent, AgentType
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

template = """
以下是一段AI机器人和人类的对话:
{chat_history}
根据输入和上面的对话记录写一份总结
输入:{input}
"""

prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=template
)

summaryMemory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

readonlyMemory = ReadOnlySharedMemory(memory=summaryMemory)

summary_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=readonlyMemory,
)

# 搜索工具
search = SerpAPIWrapper(
    serpapi_api_key="f1fd465147cd94025287abbe521d75c65e63471dd497c0c9903d5034d3e9e7ba")


# 总结工具
def summary_chain_fun(history):
  print("开始总结")
  print("输入历史为:" + history)
  summary_chain.run(history)


tools = [
  Tool(
      name="Search",
      func=search.run,
      description="需要了解实时信息的时候或者你不知道的事情的时候可以使用的搜索工具"
  ),
  Tool(
      name="Summary",
      func=summary_chain_fun,
      description="当你需要总结一段对话时可以使用这个工具,输入必须为字符串,只在必要时使用"
  )
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={
      # 90-102 无法修改 之前的叫前缀 之后的叫后缀 都可以修改
      "prefix": "尽你所能回答以下问题。您可以访问以下工具",
      # 添加历史会话
      "suffix": """Begin!
        {chat_history}
        Question: {input}
        Thought:{agent_scratchpad}
        """,
      "agent_scratchpad": MessagesPlaceholder("agent_scratchpad"),
      "input": MessagesPlaceholder("input"),
      "chat_history": MessagesPlaceholder("chat_history")
    }
)

print(agent_chain.agent.llm_chain.prompt.template)

# ---------------------print打印以下内容------------------------------------

# 'Answer the following questions as best you can. You have access to the following tools:
#
# Search(query: str, **kwargs: Any) -> str - 需要了解实时信息的时候或者你不知道的事情的时候可以使用的搜索工具
# Summary(history) - 当你需要总结一段对话时可以使用这个工具,输入必须为字符串,只在必要时使用
#
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [Search, Summary]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
#
# Begin!
#
# Question: {input}
# Thought:{agent_scratchpad}'


print(agent_chain.run("美国总统现在是谁"))
print(agent_chain.run("他的妻子是谁"))
print(agent_chain.memory.buffer)
print(agent_chain.run(input = "我们都聊了什么"))