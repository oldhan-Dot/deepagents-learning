



import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from travel_agents.config import llm


async def build_ticket_agent():
    # langchain提供链接mcp服务器的adapter
    client = MultiServerMCPClient(
        {
            "12306-mcp": {
              "transport": "http",
              "url": "https://mcp.api-inference.modelscope.net/0385a30805024e/mcp"
            }
        }
    )
    ticket_tools = await client.get_tools()

    ticket_agent = {
        "name": "ticket_agent",
        "model": llm,
        "description": "根据问题来对车票的车次、票价进行查询",
        "system_prompt": """你是一名12306车票规划助手。
                        你只负责车次查询、票价分析、直达/中转建议。

                        规则：
                        1. 只要用户已经给出出发地、目的地、出行日期中的关键信息，就优先调用12306相关工具查询
                        2. 如果缺少出行日期，先调用当前日期工具，再按“近期出行”给出默认规划
                        3. 如果缺少出发地，不要直接停止；先给出“待补充出发地后可精确查票”的说明，同时尽量补充目的地车站和交通预算建议
                        4. 如预算敏感，优先给出低价方案；如时间敏感，优先给出省时方案
                        5. 不做真实购票，只做查询与建议
                        6. 输出必须包含：票务状态、推荐方案、预算提示、还需补充的信息。""",
        "tools":ticket_tools
    }
    return ticket_agent


ticket_agent = asyncio.run(build_ticket_agent())