"""
测试4_6：使用CompiledSubAgent实现三级嵌套 SubAgent

层级：
CEO -> CTO -> CODER
实现方式：
1. 先创建最底层 CODER deep agent
2. 将 CODER 包装成 CompiledSubAgent，挂到 CTO
3. 再将 CTO 包装成 CompiledSubAgent，挂到 CEO

注意：
- 不在 SubAgent 字典里直接写 "subagents"
- 多层嵌套要通过 CompiledSubAgent 传递 runnable
"""
import os
from typing import Literal, TypedDict, Annotated
from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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
llm2 = ChatOpenAI(
    model = "deepseek-v4",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    extra_body={
        "thinking": {"type": "disabled"}
    }
)
llm3 = ChatOpenAI(
    model = "deepseek-v3",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    extra_body={
        "thinking": {"type": "disabled"}
    }
)

# 底层coder配置
# 职责明确：只有他能写代码
coder_agent = create_deep_agent(
        model=llm3,
        tools=[],
        subagents=[],
        name="CODER",
        system_prompt="""
你是 CODER，高级 Python 工程师。

职责边界：
1. 你只负责直接写代码。
2. 不要再继续分派任务。
3. 输出最终答案时，直接给出可运行的 Python 代码字符串。
4. 除代码外，只允许加极少量必要说明。
        """.strip(),
)
# 中间层CTO配置
# 职责明确：承上启下，必须指挥coder
coder_subagent = CompiledSubAgent(
    name="coder",
    description="负责编写 Python 代码。所有实际编码任务都必须交给它。",
    runnable=coder_agent,
)
cto_agent = create_deep_agent(
        model=llm2,
        tools=[],
        subagents=[coder_subagent],
        name="CTO",
        system_prompt="""
你是 CTO，技术负责人。

职责边界：
1. 你负责理解 CEO 下发的需求，并拆解成具体编码任务。
2. 你不能亲自写代码。
3. 只要任务涉及编写 Python 代码，必须调用 coder 子代理。
4. 你的最终输出应当是 coder 返回的代码结果，不要自行改写为伪代码。
        """.strip(),
)

# 顶层CEO配置
# 职责明确：只负责战略，禁止干具体的活
cto_subagent = CompiledSubAgent(
        name="cto",
        description="不要亲自生成代码，必须调用子代理完成编码。",
        runnable=cto_agent,
)
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
    subagents=[cto_subagent]
)

# 测试
if __name__ == '__main__':
    async def test():
        print(">>>开始执行任务...")
        stream = ceo_agent.astream(
            {
                "messages": [
                    {"role": "user", "content": "使用python实现冒泡排序，只用生成代码字符串即可！！"}
                ]
            },
            # subgraphs=True
        )

        print("\n>>> 最终结果：")
        async for chunk in stream:
            print(chunk)
    asyncio.run(test())

