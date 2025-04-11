from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import RouterChain, LLMRouterChain, \
  MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import \
  MULTI_PROMPT_ROUTER_TEMPLATE
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import SecretStr

llm = ChatTongyi(model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                 streaming=True)

physics_prompt = ChatPromptTemplate.from_messages([("human", "您是一位非常聪明的物理教授,\n"
                                                             "您擅长以简洁易懂的方式回答物理问题，\n"
                                                             "当您不知道问题答案的时候，您会坦率承认不知道。\n"
                                                             "下面是一个问题:{input}")])
math_prompt = ChatPromptTemplate.from_messages([("human", "您是一位非常聪明的数学教授,\n"
                                                          "当您不知道问题答案的时候，您会坦率承认不知道。\n"
                                                          "您特别会分解问题 然后再组合起来解答"
                                                          "下面是一个问题:{input}")])

prompt_infos = [
  {
    "name": "physics",
    "description": "擅长回答物理问题",
    "prompt_template": physics_prompt
  },
  {
    "name": "math",
    "description": "擅长回答数学问题",
    "prompt_template": math_prompt
  }
]

destination_chain = {}
for info in prompt_infos:
  name = info["name"]
  prompt_template = info["prompt_template"]
  chain = LLMChain(
      llm=llm,
      prompt=info["prompt_template"]
  )
  destination_chain[name] = chain

default_chain = ConversationChain(
    llm=llm,
    output_key="text",
    verbose=True
)
description = ""
for prompt in prompt_infos:
  description += prompt["name"] + ":" + prompt["description"] + "\n"

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=description)
print(router_template)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(
    # 条件chain
    router_chain=router_chain,
    # 名字-条件chain
    destination_chains=destination_chain,
    # 默认chain
    default_chain=default_chain,
    verbose=True
)
print(chain.invoke({"input":"牛顿有几个定律"}))