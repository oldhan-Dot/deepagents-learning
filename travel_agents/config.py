#加载env配置
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

#初始化模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
