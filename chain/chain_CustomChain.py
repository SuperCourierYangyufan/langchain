from typing import Dict, Any, Optional, List

from langchain.chains.base import Chain
from langchain_community.chat_models import ChatTongyi
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import SecretStr
from langchain_core.prompts.prompt import PromptTemplate


class myCustomChain(Chain):
  prompt: BasePromptTemplate
  llm: BaseLanguageModel
  out_key: str = "text"

  @property
  def input_keys(self) -> List[str]:
    return self.prompt.input_variables

  @property
  def output_keys(self) -> List[str]:
    return [self.out_key]

  # 运行链
  def _call(self, inputs: Dict[str, Any],
      run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[
    str, Any]:
    # 模板和参数汇总
    prompt_value = self.prompt.format_prompt(**inputs)
    # 大模型执行
    response = self.llm.generate_prompt(
        [prompt_value],
        # 判断是否有回调,有回调执行
        callbacks=run_manager.get_child() if run_manager else None
    )
    if run_manager:
      run_manager.on_text("执行了回调")
    #   返回输出
    return {self.out_key: response.generations[0][0].text}

  @property
  def _chain_type(self) -> str:
    return "wiki_article_chain"


chain = myCustomChain(prompt=PromptTemplate(
    template="写一篇关于{topic}的维基百科形式的文章",
    input_variables=["topic"]
), llm=ChatTongyi(model="deepseek-v3",
                  api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"),
                  streaming=True))

result = chain.invoke({"topic":"降本增效"})
print(result)