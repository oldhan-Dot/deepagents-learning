
"""
测试4_4: 通过response_format配置给子智能体指定输出数据的格式
"""
import os
from typing import Literal, TypedDict, Annotated
from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient

#加载env配置
load_dotenv()

#初始化模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    extra_body={
        "thinking": {"type": "disabled"}
    }
)

#创建Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def internet_tavily(
        query:str,# 搜索内容
        max_results:int=5,# 最大返回结果数量
        topic:Literal["general", "news", "finance"] = "general",# 搜索主题
        include_raw_content:bool=False,# 是否返回原始内容 False为精简内容
):
    """
    使用tavily进行网络搜索
    :param query: 搜索内容
    :param max_results: 最大返回结果数量
    :param topic: 搜索主题
    :param include_raw_content: 是否返回原始内容
    :return: 搜索结果
    """
    print(f"正在搜索，query={query}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,

    )


#自定义输出格式
class WeatherResponse(BaseModel):
    city:str = Field(description="城市名称")
    temperature:int = Field(description="温度")
    condition:str = Field(description="天气状况")


sub_agent={
    "name":"researcher",
    "description":"你是一个强大的助手",
    "system_prompt":"通过internet_search工具搜索结果",
    "tools":[internet_tavily],
    "response_format":WeatherResponse  # 指定格式后的数据类型

}
#设置主智能体，设置子智能体
main_agent = create_deep_agent(
    model = llm,
    system_prompt="你是一个强大的管家，你不要亲自处理任务，将任务交给researcher子智能体处理",
    subagents=[sub_agent],
    # response_format = WeatherResponse  # 指定格式后的数据类型
)
print("----开始执行任务")
result = main_agent.invoke(
    {"messages": [
    {"role": "user", "content": "武汉今天的天气怎么样？以json的格式返回数据"}]}

)

print(f"最终结果：{result['messages'][-1].content}")