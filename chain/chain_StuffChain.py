# 最常见的文档链 将文档塞进Prompt中,为LLM回答提供上下文,适合小文档场景
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr

loader = PyPDFLoader("E:\\code\\langchain\\文件示例\\激活文档.pdf")
prompt = PromptTemplate.from_template(template="对以下文字做总结:{text}")
llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
# llm_chain = RunnableSequence.from_functions([prompt, llm])
stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="text",
)
docs = loader.load()
print(stuff_chain.invoke({"text":docs}))
