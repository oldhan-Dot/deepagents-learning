from deepagents import  CompiledSubAgent
from langchain.agents import create_agent

from travel_agents.config import llm

summary_agent = create_agent(
    model=llm,
    system_prompt="""你是一名旅行方案汇总助手。
                    你不负责外部查询，只负责整理结果。
                    
                    规则：
                    1. 将景点方案与车票方案合并
                    2. 输出最终推荐方案、备选方案、预算估算
                    3. 按“需求摘要、景点建议、车票建议、预算、行程表、注意事项”结构输出
                    4. 保持简洁，不重复原始数据""",
)

manager_sum_agent = CompiledSubAgent(
    name="summary_agent",
    description="负责管理summary_agent来汇总其他两个智能体搜索的知识汇总",
    runnable=summary_agent
)