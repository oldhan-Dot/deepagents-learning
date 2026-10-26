"""
测试1：deepagents的第一个测试例子
"""
import asyncio
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
        query,max_results=max_results,include_raw_content=include_raw_content,topic=topic
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
async def query_astream(query:str):
    async for chunk in agent.astream({"messages":[{"role":"user","content":query}]}):
            #print(chunk)
            for key,value in chunk.items():
                if key not in ["tools","model"]:
                    continue #
                #得到当前message对象
                message = value["messages"][0]
                if key == "model":#模型处理返回
                    tool_calls = message.tool_calls
                    if tool_calls:#准备去调用工具
                        for tool_call in tool_calls:
                            name=tool_call["name"]
                            if name == "task":#准备调用子智能体(task tool)
                                args = tool_call["args"]
                                print(f"---模型返回，准备调用智能体[{args["subagent_type"]}],传入的参数{args["description"]}")
                    else:#返回最终结果
                        print(message.content)
                elif key =="tools":#工具调用返回
                    name = message.name
                    #子智能体的返回 task tool
                    if name =="task":
                        print(f"---子智能执行的返回, 内容: {message.content}")
                    # tool的返回
                    else:
                        print(f"---工具执行的返回，内容：{message.content}")


# 智能体3次执行是串行执行
async def test_batch_astream():
    await query_astream("AI最新资讯")
    await query_astream("agent最新资讯")
    await query_astream("大模型最新资讯")

# 智能体3次执行是并行执行
async def test_batch_astream2():
    cor1 =  query_astream("AI最新资讯")
    cor2 = query_astream("agent最新资讯")
    cor3 = query_astream("大模型最新资讯")
    await asyncio.gather(cor1, cor2, cor3)
if __name__ == '__main__':
    # asyncio.run(query_astream("今天武汉天气怎么样?"))
    # asyncio.run(test_batch_astream())
    asyncio.run(test_batch_astream2())
