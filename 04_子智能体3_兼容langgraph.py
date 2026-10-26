"""
  测试4_3: 兼容langgraph子智能体格式
"""

import os
from typing import Literal, TypedDict, Annotated

from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from tavily import TavilyClient

#加载env配置
load_dotenv()

#初始化模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

#创建Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
def internet_tavily(
        query:str,
        max_results:int=5,
        topic:Literal["general", "news", "finance"] = "general",
        include_raw_content:bool=False,
):
    """使用Tavily进行联网搜索"""
    print(f"正在搜索，query={query}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
#创建langgraph的状态图
class MyState(TypedDict):
    #注解messages的类型为list，messages的数据在多个节点之间是累加的方式
    messages:Annotated[list,add_messages] #add_messages后面状态为累加的方式
def process_node(state:MyState):
    print("----开始处理节点")
    query = state["messages"][-1].content
    result = internet_tavily(query)

    return {"messages":[AIMessage(f"经过节点处理后的结果：原数据为：{result}")]}

state_graph = StateGraph(state_schema=MyState)
state_graph.add_node("process_node", process_node)
state_graph.add_edge(START, "process_node")
state_graph.add_edge("process_node", END)
compiled_graph = state_graph.compile()
#将langchain的单智能体转换为深度代理的子智能体
custom_agent = CompiledSubAgent(
    name= "weather_subagent",
    description="你是一个天气查询助手的子智能体,自己返回天气信息",
    runnable=compiled_graph
)
#设置主智能体，设置子智能体
main_agent = create_deep_agent(
    model = llm,
    tools=[],
    system_prompt="你是一个智能助手，所有的业务动作，你不要亲自做，都找weather_subagent子智能体进行处理！",
    subagents=[custom_agent ]
)
print("----开始执行任务")
result = main_agent.invoke(
    {"messages": [
    {"role": "user", "content": "武汉今天的天气怎么样？"}]}

)

print(f"最终结果：{result['messages'][-1].content}")