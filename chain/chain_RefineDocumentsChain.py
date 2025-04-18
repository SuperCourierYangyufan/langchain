# 首先从第一段文档生成一个“初始回答”。然后依次遍历其他文档，每次让模型“改写/改进”上一个回答。最终形成一个“精炼”的总结/答案。所以它是一种增量式总结机制，而不是一次性处理所有文档。
from gitdb.fun import chunk_size
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from pydantic import SecretStr

# 理论上应该是多个文件
loader = PyPDFLoader("E:\\code\\langchain\\文件示例\\激活文档.pdf")
docs = loader.load()
text_split = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=30,
    chunk_overlap=5
)
split_docs = text_split.split_documents(docs)
init_prompt = PromptTemplate.from_template(
    "给定以下内容,生成一个回答:{text}")
refine_prompt = PromptTemplate.from_template("已有的答案如下：\n{existing_answer}\n"
                                             "给定以下文档片段，你可以选择保持原样，或者进行改进：\n{text}")
llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
refine_chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=init_prompt,
    refine_prompt=refine_prompt,
    initial_response_name="existing_answer",
    verbose=True,
)

result = refine_chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
print(result)
