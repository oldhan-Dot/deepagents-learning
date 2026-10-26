"""
    测试4_5: 字典形式的嵌套子agent  (不支持)
    原因：字典形式的子agent中没有处理subagents配置
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

# 底层CODER agent
# 职责明确：只有他能写代码
coder_agent = {
    "name": "CODER",
    "description": "高级Python工程师，他是唯一有权限编写具体代码的人。",
    "system_prompt": "你是一个高级Python工程师。你的职责是接收具体的编码任务并实现它。",
    "tools": [] # coder拥有默认的文件操作工具
}

# 中间层CTO agent
# 职责明确：承上启下，必须指挥coder
cto_config = {
    "name": "CTO",
    "description": "技术总监，负责将战略需求转化为技术任务并分配给工程师。指挥Coder写代码的！",
    # 关键修改：明确告诉CTO不要自己写代码，必须找CODER
    "system_prompt": """
        你是技术总监。
        注意：你没有编写代码的权限！
        你的职责是：
        1. 分析 CEO 的需求。
        2. 设计技术方案。
        3. 调用 'Coder' 子代理来完成具体的代码编写工作。
    """,
    "tools": [], # coder拥有默认的文件操作工具
    "subagents": [coder_agent] # 子智能体中没有一个配置，强写（底层不识别）
}

# 顶层CEO agent
# 职责明确：只负责战略，禁止干具体的活
ceo_agent = create_deep_agent(
    model=llm,
    name="CEO",
    # 关键修改：明确告诉CEO不要自己动手，必须找CTO
    system_prompt="""
        你是CEO，负责公司战略决策。
        注意：你严禁直接编写代码或操作文件！
        你必须将所有技术相关的开发任务委派给 'CTO' 处理。
        你的工作是验收 CTO 提交的结果。
    """,
    subagents=[cto_config]
)

if __name__ == '__main__':
    async def test():
        print(">>>开始执行任务...")
        stream = ceo_agent.astream(
            {
                "messages": [
                    {"role": "user", "content": "使用python实现冒泡排序，只用生成代码字符串即可！！"}
                ]
            }
        )

        print("\n>>> 最终结果：")
        async for chunk in stream:
            print(chunk)


    asyncio.run(test())
