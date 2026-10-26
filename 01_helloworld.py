"""
测试1：deepagents的第一个测试例子
"""
import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

#创建tavily客户端
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

#定义工具
@tool
def internet_search(
    query:str,#用户提问
    max_results:int =5 ,#返回内容最大数量
    topic:Literal["general","news","finance"]="general",
    include_raw_content:bool=False,#是否要原始数据
):
    """ 使用Tavily进行网络搜索"""
    print("执行internet_search工具")
    return tavily_client.search(
        query,max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )

#初始化模型
llm = ChatOpenAI(
    model="deepseek-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

#创建深度智能体
agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
  system_prompt = """"你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。
"""
)
result = agent.invoke({"messages":[{"role":"user","content":"今天武汉天气"}]})
print(result)
print(result["messages"][-1].content)