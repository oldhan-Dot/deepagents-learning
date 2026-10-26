"""
测试4_1: 子智能体的基本使用  => 基于字典定义子agent
创建一个主智能体，它拥有三个助手：
    1.  **天气助手**：查询天气（固定返回“晴朗”）。
    2.  **计算助手**：处理数学问题。
    3.  **翻译助手**：负责中英互译。
"""
import os
from deepagents import create_deep_agent
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
#加载env配置
load_dotenv()

#初始化模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

weather_llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)



#1.  **天气助手**：查询天气（固定返回“晴朗”）。
weather_agent = {
    "name":"weather_helper",
    "description":"你是一个天气助手可以查询天气",#给主智能体看
    "system_prompt":"你是一个天气助手, 直接返回天气信息:'今天天气晴，温度：20度",
    "model":weather_llm #子智能体特有的模型 （没指定则为默认模型-兜底模型）
}
    #计算助手子agent
math_agent = {
    "name":"math_helper",
    "description":"你是一个数学计算高手",
    "system_prompt":"你是一个严谨的数学计算高手，回答用户的所有计算问题",
    "tools":[]
}
# 翻译助手子agent
translate_agent = {
    "name": "translate_helper",
    "description": "一个中英相互翻译助手",
    "system_prompt": "你是一个中英想到翻译助手，如果是中文翻译为英文，反之翻译为中文",
    "tools": []
}
#创建深度代理
agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="你是一个强大的管家，不要亲自执行任务，将任务交给合适子智能完成",
    subagents=[weather_agent,math_agent,translate_agent]# 配置子智能体
)

def query_stream(query:str):
    for chunk in agent.stream({
        "messages":[{
            "role":"user","content":query
        }]
    }):
        #print(chunk)
        for key,value in chunk.items():
            if key in ["model","tools"]:
                message = value["messages"][0]
                if key =="model":#模型处理返回
                    tool_calls = message.tool_calls #要调用的工具列表
                    if tool_calls: #tool_calls有值，说明准备调用工具
                        for tool_call in tool_calls:
                            print(f"--模型返回，决定调用工具{tool_call["name"]},参数为{tool_call["args"]}")
                    else:#如果没有说明是大模型返回的最终结果
                        print(f"模型返回，最终结果：{message.content}")
                elif key =="tools":#工具处理返回
                    name = message.name #工具名字
                    content = message.content #结果数据
                    print(f"--工具[{name}]执行的结果为:{content}")
if __name__ == '__main__':
    query_stream("今天武汉天气怎么样? 再计算123+234-99等于几？，同时将'好好学习，天天向上'翻译为英文")