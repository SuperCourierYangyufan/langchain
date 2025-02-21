from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

examples = [
  {
    "question": "谁的寿命更长，穆罕默德·阿里还是艾伦·图灵?",
    "answer":
      """
    这里需要跟进问题吗:是的。
    跟进:穆罕默德·阿里去世时多大?
    中间答案:穆罕默德·阿里去世时74岁
    跟进:艾伦·图灵去世时多大?
    中间答案:艾伦·图灵去世时41岁
    所以最终答案是:穆罕默德·阿里
    """
  },
  {
    "question": "craigslist的创始人是什么时候出生的?",
    "answer":
      """
    这里需要跟进问题吗:是的。
    跟进:craigslist的创始人是谁?
    中间答案:craigslist由craig Newmark创立。
    跟进:Craig Newmark是什么时候出生的?
    中间答案:Craig Newmark于1952年12月6日出生。
    所以最终答案是:1952年12月6日
    """
  }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    # 文字转向量 使用的openapi的 需要对应key
    OpenAIEmbeddings(),
    # 内存级别向量数据库
    Chroma,
    # 生成的实例数
    k=1
)

question = "穆罕默德·阿里还是艾伦·图灵那个先出生?"
response = example_selector.select_examples({"question": question})
print(response)