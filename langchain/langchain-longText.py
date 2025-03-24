# 长上下文精度问题
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_transformers import LongContextReorder
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import SecretStr

embeddings=DashScopeEmbeddings(
        dashscope_api_key="sk-7dfe20a081554d3e896c7044ed951b0a",
        model="text-embedding-v3")
llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)

text = [
    "国内油价迎年内第十涨，92号汽油每升上调0.12元",
    "巴以冲突持续升级，已致双方超2300人死亡",
    "华为Mate60 Pro+开售，搭载自研麒麟9100S芯片",
    "央行下调MLF利率10个基点，释放稳增长信号",
    "我国成功发射遥感三十九号卫星，实现一箭三星",
    "台风'小犬'逼近福建沿海，启动防台风Ⅳ级响应",
    "诺贝尔经济学奖揭晓，美学者研究劳动市场获奖",
    "杭州亚运会闭幕式举行，中国代表团金牌破200枚",
    "特斯拉Model3焕新版上市，售价25.99万元起",
    "全国秋粮收获过半，预计产量再创历史新高",
    "欧盟启动对中国电动汽车反补贴调查",
    "电影《志愿军》票房破6亿，领跑国庆档期",
    "北京环球影城'冬季假日'主题活动11月启幕",
    "我国首架氢能源飞机试飞成功",
    "日本第二轮核污染水排海，中方严正交涉",
    "三季度CPI同比上涨0.2%",
    "新疆塔里木盆地新发现亿吨级油气田",
    "中国女足奥预赛名单公布，王霜领衔",
    "苹果推送iOS17.0.3更新，修复异常发热",
    "A股三大指数反弹，新能源汽车板块领涨"
]

chroma = Chroma.from_texts(text,embeddings).as_retriever(
    # 返回最相关的信息 条数
    search_kwargs  = {"k":5}
)

result = chroma.get_relevant_documents("汽油相关的信息")
print(result)

# 大模型对头尾的问题识别率更高,正常向量数据库返回是根据相关性高->低排序
# 用LongContextReorder().transform_documents  改为高->低->高
result =  LongContextReorder().transform_documents(result)
print(result)