import asyncio

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from travel_agents.config import llm


async def build_map_agent():
    # langchain提供链接mcp服务器的adapter
    client = MultiServerMCPClient(
        {
            "amap-maps": {
              "transport": "streamable_http",
              "url": "https://mcp.api-inference.modelscope.net/1f9fc2b59d5942/mcp"
            }
        }
    )
    map_tools = await client.get_tools()





    map_agent = {
        "name":"map_agent",
        "description":"根据问题来查询地点路线等信息",
        "model":llm,
        "system_prompt":"""你是高德地图 MCP 助手。
            只回地点、路线、周边、距离、POI 详情和天气问题。
            
            优先使用高德地图 MCP 工具；缺少城市、起点、终点、关键词或范围时先追问。
            不得编造地址、距离、耗时、评分、电话或营业时间。
            工具返回为空或不完整时，说明未查询到相关信息，并提示需要补充的内容。
            非地图类问题不调用地图工具。
            
            输出时只给结论：名称、地址、距离、耗时、路线方案或天气要点；不暴露原始 JSON。""",
        "tools":map_tools
    }
    return map_agent



map_agent = asyncio.run(build_map_agent())