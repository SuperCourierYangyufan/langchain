# 示例选择器
# 根据长度要求智能选择示例
# 根据输入相似度选择示例(最大边际相关性)
# 根据输入相似度选择示例(最大余弦相似度)
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import LengthBasedExampleSelector, \
  MaxMarginalRelevanceExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr


llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

examples = [
  {"input": "好", "output": "坏"},
  {"input": "天晴", "output": "下雨"},
  {"input": "上", "output": "下"},
  {"input": "左", "output": "右"},
  {"input": "肥胖", "output": "瘦小"},
  {"input": "美丽的", "output": "丑陋的"},
  {"input": "高", "output": "矮"},
  {"input": "大", "output": "小"},
  {"input": "早", "output": "晚"},
  {"input": "强", "output": "弱"},
  {"input": "聪明", "output": "愚笨"},
  {"input": "快乐", "output": "悲伤"},
  {"input": "快", "output": "慢"},
  {"input": "多", "output": "少"},
  {"input": "明亮", "output": "昏暗"},
  {"input": "新", "output": "旧"},
  {"input": "强烈", "output": "微弱"},
  {"input": "黑", "output": "白"},
  {"input": "宽", "output": "窄"},
  {"input": "厚", "output": "薄"},
  {"input": "清晰", "output": "模糊"},
  {"input": "富有", "output": "贫穷"},
  {"input": "温暖", "output": "寒冷"},
  {"input": "宽容", "output": "苛刻"},
  {"input": "成功", "output": "失败"},
  {"input": "喜欢", "output": "讨厌"},
  {"input": "聪明", "output": "愚蠢"},
  {"input": "勇敢", "output": "胆小"},
  {"input": "真", "output": "假"},
  {"input": "高兴", "output": "难过"},
  {"input": "干净", "output": "脏"},
  {"input": "安静", "output": "嘈杂"},
  {"input": "真诚", "output": "虚伪"},
  {"input": "高贵", "output": "卑贱"},
  {"input": "开朗", "output": "阴郁"},
  {"input": "干燥", "output": "潮湿"},
  {"input": "年轻", "output": "年老"},
  {"input": "漂亮", "output": "丑陋"},
  {"input": "健康", "output": "生病"},
  {"input": "平稳", "output": "动荡"},
  {"input": "好奇", "output": "无聊"},
  {"input": "爱", "output": "恨"},
  {"input": "平静", "output": "激烈"},
  {"input": "成功", "output": "失败"},
  {"input": "正面", "output": "负面"},
  {"input": "单纯", "output": "复杂"},
  {"input": "聪明", "output": "傻"},
  {"input": "坚强", "output": "脆弱"},
  {"input": "快速", "output": "缓慢"},
  {"input": "上升", "output": "下降"},
  {"input": "厚重", "output": "轻盈"},
  {"input": "激烈", "output": "平和"},
  {"input": "温和", "output": "激烈"},
  {"input": "光亮", "output": "黑暗"},
  {"input": "有趣", "output": "无聊"},
  {"input": "锋利", "output": "钝"},
  {"input": "清澈", "output": "浑浊"},
  {"input": "安静", "output": "吵闹"},
  {"input": "亲切", "output": "冷淡"},
  {"input": "直接", "output": "间接"},
  {"input": "豪华", "output": "简朴"},
  {"input": "正义", "output": "邪恶"},
  {"input": "重", "output": "轻"},
  {"input": "丰富", "output": "贫乏"},
  {"input": "透明", "output": "不透明"},
  {"input": "有序", "output": "无序"},
  {"input": "湿润", "output": "干燥"},
  {"input": "平衡", "output": "不平衡"},
  {"input": "清爽", "output": "闷热"},
  {"input": "寒冷", "output": "温暖"},
  {"input": "高明", "output": "拙劣"},
  {"input": "清晰", "output": "模糊"},
  {"input": "现代", "output": "古老"},
  {"input": "新鲜", "output": "腐烂"},
  {"input": "甜", "output": "苦"},
  {"input": "自信", "output": "自卑"},
  {"input": "丰富", "output": "贫乏"},
  {"input": "长", "output": "短"},
  {"input": "光滑", "output": "粗糙"},
  {"input": "小心", "output": "粗心"},
  {"input": "安全", "output": "危险"},
  {"input": "和谐", "output": "冲突"},
  {"input": "微弱", "output": "强烈"},
  {"input": "充实", "output": "空虚"},
  {"input": "勇气", "output": "恐惧"},
  {"input": "生动", "output": "枯燥"},
  {"input": "复杂", "output": "简单"},
  {"input": "美好", "output": "糟糕"},
  {"input": "成熟", "output": "幼稚"},
  {"input": "有力", "output": "无力"},
  {"input": "宁静", "output": "喧闹"},
  {"input": "纯洁", "output": "污秽"},
  {"input": "繁忙", "output": "闲暇"},
  {"input": "喜爱", "output": "厌恶"},
  {"input": "安定", "output": "动荡"},
  {"input": "宽容", "output": "狭隘"},
  {"input": "安静", "output": "喧嚣"},
  {"input": "正直", "output": "歪曲"},
  {"input": "朴素", "output": "华丽"},
  {"input": "开朗", "output": "沉闷"},
  {"input": "渴望", "output": "冷漠"},
  {"input": "鲜艳", "output": "黯淡"},
  {"input": "进步", "output": "退步"},
  {"input": "安宁", "output": "动乱"},
  {"input": "丰富", "output": "单一"},
  {"input": "强烈", "output": "微弱"},
  {"input": "豪华", "output": "简陋"},
  {"input": "甜美", "output": "苦涩"},
  {"input": "得意", "output": "沮丧"},
  {"input": "温暖", "output": "寒冷"},
  {"input": "热情", "output": "冷淡"},
  {"input": "胜利", "output": "失败"},
  {"input": "庄重", "output": "轻浮"},
  {"input": "成熟", "output": "未熟"},
  {"input": "浓烈", "output": "淡"},
  {"input": "稳定", "output": "不稳定"},
  {"input": "欢喜", "output": "忧愁"},
  {"input": "激动", "output": "冷静"},
  {"input": "谦虚", "output": "自大"},
  {"input": "开放", "output": "封闭"},
  {"input": "理智", "output": "感性"},
  {"input": "利", "output": "弊"},
  {"input": "聪慧", "output": "愚笨"},
  {"input": "温和", "output": "暴躁"},
  {"input": "精致", "output": "粗糙"},
  {"input": "清洁", "output": "肮脏"},
  {"input": "冷静", "output": "激动"},
  {"input": "浓密", "output": "稀疏"},
  {"input": "昂贵", "output": "便宜"},
  {"input": "正面", "output": "负面"},
  {"input": "勇敢", "output": "懦弱"},
  {"input": "直接", "output": "间接"},
  {"input": "清澈", "output": "浑浊"},
  {"input": "辉煌", "output": "黯淡"},
  {"input": "健康", "output": "病态"},
  {"input": "茂盛", "output": "枯萎"},
  {"input": "宽敞", "output": "狭窄"},
  {"input": "宽大", "output": "紧身"},
  {"input": "激烈", "output": "平淡"},
  {"input": "光明", "output": "黑暗"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词:{input}\n反义词:{output}"
)

# 长度选择器 默认\n 换行
length_selector = LengthBasedExampleSelector(
    # 提示词示例
    examples=examples,
    # 提示词模板
    example_prompt=example_prompt,
    # 格式化后提提示词最大长度,提示词过长后,默认只截取25个提示词的大小
    max_length=25,
)

# 语义相关选择器
marginal_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 传入示例组
    examples=examples,
    # embeddings
    embeddings=OpenAIEmbeddings(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
        model="text-embedding-v3"),
    # 使用向量数据库
    vectorstore_cls=Chroma,
    # 结果条数
    k=2
)

# 提示词模板,小样本
dynamic_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    # example_selector=length_selector,
    example_selector=marginal_selector,
    prefix="给出原词的反义词",
    suffix="原词:{input},\n反义词:",
    input_variables=["input"]
)

print(dynamic_prompt.format(input="辉煌"))

print(llm.invoke(dynamic_prompt.format(input="辉煌")))
