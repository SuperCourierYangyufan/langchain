# 数据转换
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.chains.transform import TransformChain
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr

# 创建大模型实例
llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)

# 转换函数：将输入转为小写
def transform_func(input: dict) -> dict:
    text = input["text"]
    return {"lower_text": text.lower()}  # 注意，这里我们使用不同的key名：lower_text

# 定义 TransformChain
transform = TransformChain(
    input_variables=["text"],           # 输入来自原始输入
    output_variables=["lower_text"],    # 输出是新的变量名
    transform=transform_func
)

# 定义 Prompt 模板
prompt = PromptTemplate.from_template(
    """告诉我这个单词是大写还是小写？
    单词: {lower_text}
    """
)

# 创建 LLMChain，注意 prompt 的变量是 lower_text
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key="final_output"  # 避免默认使用 "text" 导致冲突
)

# 定义 SequentialChain，连接两个子链
seq_chain = SequentialChain(
    chains=[transform, llm_chain],
    input_variables=["text"],
    output_variables=["final_output"],  # 只输出最终结果
    verbose=True
)

# 执行链
print(seq_chain.run({"text": "APPLE"}))
