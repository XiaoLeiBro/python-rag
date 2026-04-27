# 使用 ChatOpenAI 对接百炼 OpenAI 兼容接口
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# ---- 1. 简单调用 ----
# res = model.invoke("你是谁啊，什么大模型")
# print(res.content)

# ---- 2. 流式输出 ----
# for chunk in model.stream("你是谁啊，什么大模型"):
#     print(chunk.content, end="", flush=True)

# ---- 3. 多轮对话（带系统提示词） ----
messages = [
    # SystemMessage(content="你是一个乐于助人的助手，请用中文回答。"),
    # HumanMessage(content="用一句话解释什么是向量数据库"),
    SystemMessage(content="你是一名边塞诗人。"),
    HumanMessage(content="用人类的语言给我写一首诗"),
]
# res = model.invoke(messages)
# print(res.content)
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
