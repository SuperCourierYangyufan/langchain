from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import SecretStr

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-7dfe20a081554d3e896c7044ed951b0a",
    model="text-embedding-v3")

news_list = [
  "国内经济持续向好，市场信心增强。",
  "科技创新成果显著，多项技术获突破。",
  "国际局势紧张，多国展开外交对话。",
  "环保政策加码，绿色能源受关注。",
  "教育改革持续推进，新课程标准发布。",
  "医疗健康领域投资增加，行业前景看好。",
  "文化旅游产业复苏，各地推出新项目。",
  "体育赛事精彩纷呈，国家队表现亮眼。",
  "影视娱乐行业回暖，多部大片定档。",
  "互联网企业加速布局人工智能领域。",
  "房地产市场调控政策效果显现。",
  "新能源汽车销量增长，产业链受益。",
  "食品安全问题引发社会广泛关注。",
  "乡村振兴战略实施，农村面貌焕然一新。",
  "数字经济成为新的经济增长点。",
  "金融监管加强，防范系统性风险。",
  "就业形势总体稳定，创业氛围浓厚。",
  "气象部门预测今年冬季气温偏低。",
  "网络安全意识提升，防护措施加强。",
  "国际贸易摩擦加剧，企业寻求新机遇。"
]
# 缓存
fs = LocalFileStore("E://code//langchain/cache")
cache = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    fs,
    namespace=embeddings.model
)
# 向量数据库
documents = [Document(page_content=text) for text in news_list]
db = Chroma.from_documents(documents,cache)