from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, \
   RunnablePassthrough
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
  MessagesPlaceholder(variable_name="chat_history"),
  ("human", "{input}")
])
runnable = prompt | llm

chain_with_message_history = RunnableWithMessageHistory(
    runnable,
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


def summarize_message(chain_input):
  store_message = temp_chat_history.messages
  if len(store_message) == 0:
    return False
  summarize_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    (
      "user", "将上述聊天浓缩成一条摘要消息。尽可能包含多个具体细节"
    )
  ])
  chain = summarize_prompt | llm
  summary_message = chain.invoke({"chat_history": store_message})
  temp_chat_history.clear()
  temp_chat_history.add_message(summary_message)
  return True


chain_with_summarization = (RunnablePassthrough.assign(
  messages_summarized=summarize_message) | chain_with_message_history)

response = chain_with_summarization.invoke({"input": "我是谁 干嘛"},
                                {"configurable": {"session_id": "unused"}})
print(response)
print(temp_chat_history.messages)
