from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)
# 简单的基础链
first_prompt = ChatPromptTemplate.from_messages([
  ("human", "帮我把下面内容翻译成英文:{content}")
])
first_llm_chain = LLMChain(llm=llm,
                           prompt=first_prompt,
                           # 是否开启日志
                           verbose=True,
                           output_key="english_content")

second_prompt = ChatPromptTemplate.from_messages([
  ("human", "帮我一句话进行总结:{english_content}")
])
second_llm_chain = LLMChain(llm=llm,
                            prompt=second_prompt,
                            # 是否开启日志
                            verbose=True,
                            output_key="translation_content")

third_prompt = ChatPromptTemplate.from_messages([
  ("human", "智能识别下面语言是哪个国家的:{translation_content}")
])
third_llm_chain = LLMChain(llm=llm,
                           prompt=third_prompt,
                           # 是否开启日志
                           verbose=True,
                           output_key="language")

fourth_prompt = ChatPromptTemplate.from_messages([
  ("human",
   "请使用指定语言对以下内容进行回复:\n 内容:{translation_content}\n语言:{language}")
])
fourth_llm_chain = LLMChain(llm=llm,
                            prompt=fourth_prompt,
                            # 是否开启日志
                            verbose=True,
                            output_key="reply")

overall_chain = SequentialChain(
    chains=[first_llm_chain, second_llm_chain, third_llm_chain,
            fourth_llm_chain],
    verbose=True,
    input_variables=["content"],
    output_variables=["english_content", "translation_content", "language",
                      "reply"]
)
print(overall_chain(
  "近日，据《检察日报》披露：5具尸体、5米水池、24年追凶，一组数字揭开了1999年重庆出租屋电击屠杀案的骇人真相。当年，5名“棒棒”（指以给人提供劳力搬运行李货物为生的体力劳动者）被诱骗至嫌犯廖某改造的带电水池中舀水，触电身亡。因廖某冒用他人身份证，案件成悬案。2022年，警方通过现场遗留瓶身指纹锁定刚出狱的廖某。调查发现，原来，廖某梦想发明“永动机”，是想通过杀人来锻炼胆量、突破思维极限、激发所谓的创造灵感。于是他以搬货为名，将“棒棒”骗到出租屋，制造了这起电击屠杀的悲剧。最高检2023年核准追诉，认定其犯罪手段残忍、后果严重。2024年12月，重庆市五中院一审判处廖某死刑。目前案件正在二审审理期间"))
