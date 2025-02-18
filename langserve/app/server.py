from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from pydantic import SecretStr

app = FastAPI(title="我的服务器",
              version="1.0",
              description="我的描述")
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")



add_routes(app, ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))
           ,path="/openai")


add_routes(app, ChatPromptTemplate.from_template("告诉我一个关于:{topic}的笑话")
           |ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="deepseek-v3",
                 api_key=SecretStr("sk-7dfe20a081554d3e896c7044ed951b0a"))
           ,path="/openai_ext")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
