from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

prompt = ChatPromptTemplate.from_messages([
  ("system", "你是我的助手"),
  #   历史消息占位符
  MessagesPlaceholder(variable_name="history"),
  ("human", "{input}")
])
runnable = prompt | llm
# 会话历史记录
store = {}


# 定义函数 入参session_id 返回 历史记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]


# 创建一个带会话记录的Runnable
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
# 调用带会话历史记录的runnable
result = with_message_history.invoke({"input": "讲一个富含寓意的故事"}, config={
  "configurable": {"session_id": "session_1"}})
print(result)
# 同session_id 知道上文
result = with_message_history.invoke({"input": "这个故事的寓意是什么"}, config={
  "configurable": {"session_id": "session_1"}})
print(result)
# 不知道上文
result = with_message_history.invoke({"input": "这个故事的寓意是什么"}, config={
  "configurable": {"session_id": "session_2"}})
print(result)
