# pip install docx2txt,pypdf,nltk

from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, \
  UnstructuredExcelLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

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
  def embeddingAndVector(self):
    db = Chroma.from_documents(documents=self.splitText,embedding=embeddings)
    return db
  def askAndFind(self,question: str):
    db = self.embeddingAndVector()
    retriever = db.as_retriever()
    return retriever.invoke(question)


chat_doc = ChatDoc()
chat_doc.path = "../文件示例/ths安装步骤.docx"
chat_doc.splitFile()
result = chat_doc.askAndFind("ths是什么")
print(result)
