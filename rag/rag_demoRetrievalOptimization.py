# 检索优化
# pip install docx2txt,pypdf,nltk
from langchain.retrievers import MultiQueryRetriever, \
  ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, \
  UnstructuredExcelLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
import logging

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-7dfe20a081554d3e896c7044ed951b0a",
    model="text-embedding-v3")


class ChatDoc:
  def __init__(self):
    self.path = None
    self.splitText = []
    self.template = [
      ("system","你是一个THS的工程师,你会根据下面的上下文内容继续回答问题.\n "
                "你从不说自己是一个大模型AI助手"
                "上下文内容:\n"
                "{context}"),
      ("human","你好!"),
      ("ai","你好"),
      ("human","{question}")
    ]
    self.prompt = ChatPromptTemplate.from_messages(self.template)
  # 获取文件
  def getFile(self):
    path = self.path
    loaders = {
      "docx": Docx2txtLoader,
      "pdf": PyPDFLoader,
      "xlsx": UnstructuredExcelLoader
    }
    type = path.split(".")[-1]
    loaderClass = loaders.get(type)
    if loaderClass:
      try:
        loader = loaderClass(path)
        return loader.load()
      except Exception as e:
        print("加载文件报错,详情为e:{}", e)
    else:
      raise Exception("未知的类型")

  # 切割文件
  def splitFile(self):
    full_text = self.getFile()
    spliter = RecursiveCharacterTextSplitter(
        # 是否使用正则表达式
        is_separator_regex=False,
        # 切割的标识符
        separators=[',', '\n'],
        # 切分的文本块大小
        chunk_size=30,
        # 重叠区文本块的大小
        chunk_overlap=10,
        # 长度函数
        length_function=len,
        # 是否添加开始索引
        add_start_index=True
    )
    texts = spliter.split_documents(full_text)
    self.splitText = texts

  # 文本向量化
  def embeddingAndVector(self):
    db = Chroma.from_documents(documents=self.splitText, embedding=embeddings)
    return db

  # 提高精度 1. 把问题交给Llm多维度扩展
  def askAndFind(self, question: str):
    db = self.embeddingAndVector()
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm
    )
    return retriever_from_llm.invoke(question)

  # 上下文压缩  将问题压缩  提取关键词 增加精度
  def askAndFind1(self, question: str):
    db = self.embeddingAndVector()
    return ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=db.as_retriever()
    ).invoke(question)

  # MMR 最大边际相似性
  def askAndFind2(self, question: str):
    db = self.embeddingAndVector()
    retriever = db.as_retriever(search_type="mmr")
    return retriever.invoke(question)

  # 相似性打分 0.5分以上才会返回,返回3个结果
  def askAndFind3(self, question: str):
    db = self.embeddingAndVector()
    retriever = db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"score_threshold": 0.5, "k": 3})
    return retriever.invoke(question)
  # 问答
  def chatWithDoc(self, question: str):
    _content = ""
    contextList = self.askAndFind3(question)
    for context in contextList:
      _content += context.page_content
    message =  self.prompt.format_messages(context=_content,question = question)
    return llm.invoke(message)

chat_doc = ChatDoc()
chat_doc.path = "../文件示例/ths安装步骤.docx"
chat_doc.splitFile()
result = chat_doc.chatWithDoc("THS如何安装")

print(result)
