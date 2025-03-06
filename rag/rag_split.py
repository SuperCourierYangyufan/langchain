from langchain_community.document_loaders import TextLoader, DirectoryLoader, \
  PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PDFMinerLoader("C:\\Users\\Administrator\\WPSDrive\\1368392886\\WPS云盘\\学习\\阿里，字节，华为2021年度Java面试题汇总.pdf")


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

# 生成文档
text = spliter.create_documents([doc.page_content for doc in loader.load()])
print(text)
