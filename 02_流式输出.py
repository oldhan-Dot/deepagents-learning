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
def query_astream(query:str):
    for chunk in agent.stream({"messages":[{"role":"user","content":query}]}):
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
                        name=tool_calls["name"]
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


if __name__ == '__main__':
    query_astream(query="武汉天气怎么样")
