from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import asyncio

llm = ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))

# 同步
# chunks = []
# for chunk in llm.stream("天空是什么颜色"):
#   chunks.append(chunk)
#   print(chunk.content, end="|", flush=True)



# 异步
prompt = ChatPromptTemplate.from_template("给我中国根据年份 和 每年的人口数的列表,以json格式返回")
# 根据json返回
chain = prompt | llm | JsonOutputParser()
async def main():
  async for chunk in chain.astream({}):
    print(chunk, end="| \n", flush=True)
asyncio.run(main())

