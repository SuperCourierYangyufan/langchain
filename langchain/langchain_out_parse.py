# 返回信息结构化
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr, BaseModel, Field

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)

# parser = CommaSeparatedListOutputParser()
#
# prompt = PromptTemplate(
#     template="列出10个{query}:{format_instructions}",
#     input_variables=["query"],
#     partial_variables={
#         "format_instructions": parser.get_format_instructions()
#     }
# )

# 定义class
class User(BaseModel):
    name: str = Field(description="名字")
    age: int = Field(description="年龄")

    #验证名字
    def validate_name(cls, field):
        if len(field) < 2:
            raise ValueError("名字长度必须大于2")
        return field

parser = PydanticOutputParser(pydantic_object=User)

prompt = PromptTemplate(
    template= "回答用户的输入:\n{format_instructions}\n{query}\n请仅返回 JSON 格式的数据，不要包含任何其他内容或解释。",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | llm
# result = chain.invoke({"query": "2025年适合结婚的日子"})
result = chain.invoke({"query": "孙中山年龄是多少"})
print(result)
print(parser.parse(result.content))