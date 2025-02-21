from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, \
  RedisChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, \
  ConfigurableFieldSpec, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

temp_chat_history = ChatMessageHistory()
temp_chat_history.add_user_message("我是你爹")
temp_chat_history.add_ai_message("你好,爹")
temp_chat_history.add_user_message("我今天打篮球")
temp_chat_history.add_ai_message("打的怎么样")
temp_chat_history.add_user_message("100分")
temp_chat_history.add_ai_message("可以 厉害")

prompt = ChatPromptTemplate.from_messages([
  ("system", "助手"),
  #   历史消息占位符
  MessagesPlaceholder(variable_name="history"),
  ("human", "{input}")
])
runnable = prompt | llm
store = {}


def trim_message(chain_input):
  stored_message = temp_chat_history.messages
  if len(stored_message) <= 2:
    return False
  temp_chat_history.clear()
  for message in stored_message[-2:]:
    temp_chat_history.add_message(message)
  return True


chain_with_message_history = with_message_history = RunnableWithMessageHistory(
    runnable,
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

chain_with_trimming = (RunnablePassthrough.assign(
  message_trimmed=trim_message) | chain_with_message_history)

response = chain_with_trimming.invoke({"input":"我今天干嘛了"},{"configurable":{"session_id":"unused"}})
print(response)