from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, \
  ConfigurableFieldSpec
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
store = {}


def get_session_history(user_id: str,
    conversation_id: str) -> BaseChatMessageHistory:
  if (user_id, conversation_id) not in store:
    store[(user_id, conversation_id)] = ChatMessageHistory()
  return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    # 动态拓展参数,必须和get_session_history函数保持一直
    history_factory_config=[
      ConfigurableFieldSpec(
          id="user_id",
          annotation=str,
          name="User ID",
          description="用户唯一标识",
          default="",
          is_shared=True
      ),
      ConfigurableFieldSpec(
          id="conversation_id",
          annotation=str,
          name="Conversation ID",
          description="对话唯一标识",
          default="",
          is_shared=True
      )
    ]
)

# 根据函数,必须相同的用户id+对话id才能记住上文
result = with_message_history.invoke({"input": "讲一个富含寓意的故事"}, config={
  "configurable": {"user_id": "session_1","conversation_id":"1"}})
print(result)

result = with_message_history.invoke({"input": "这个故事什么寓意？"}, config={
  "configurable": {"user_id": "session_1","conversation_id":"1"}})
print(result)
result = with_message_history.invoke({"input": "这个故事什么寓意？"}, config={
  "configurable": {"user_id": "session_1","conversation_id":"1"}})
print(result)