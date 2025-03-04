# 自定义模板
# 根据函数名称,查询函数代码,并给出中文代码说明
import inspect
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import StringPromptTemplate
from langchain_core.prompts.base import FormatOutputType
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))


def hello_word():
  print("hello word")


# python包 根据函数名获取源代码
def get_source_code(function_name):
  return inspect.getsource(function_name)


prompt = """
你是一个程序员,现在给你函数名称,你会按照如下格式,输出这段代码的名称,源代码,中文解释
函数名称{function_name}
源代码{source_code}
代码解释:
"""


class CustomPrompt(StringPromptTemplate):
  def format(self, **kwargs: Any) -> FormatOutputType:
    # 获取源代码
    source_code = get_source_code(kwargs["function_name"])
    # 生成提示词模板
    return prompt.format(
        function_name=kwargs["function_name"].__name__,
        source_code=source_code
    )

template = CustomPrompt(input_variables=["function_name"])

pm = template.format(function_name=hello_word)

result = llm.invoke(pm)

print(result)
