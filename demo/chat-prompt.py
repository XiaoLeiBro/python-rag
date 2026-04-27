# 聊天提示词模板：支持植入任意数量的历史会话信息

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 创建聊天提示词模板：通过from_messages方法，创建一个支持植入任意数量历史会话信息的聊天提示词模板
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助理。"),
    MessagesPlaceholder("history"),
    ("human", "向量数据库中，向量相似度的算法都有哪些？"),
])

history_data = [
    ("human", "用一句话解释什么是向量数据库"),
    ("ai", "向量数据库是一种用于存储和检索向量的数据库。")
]

# 调用invoke方法，将历史会话数据注入模板
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
print(prompt_text)

load_dotenv()

model = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
res = model.invoke(input=prompt_text)
print(res.content)
