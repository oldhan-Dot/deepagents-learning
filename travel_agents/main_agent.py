import os

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith.types import agent

from travel_agents.config import llm
from travel_agents.ticket_agent import ticket_agent
from travel_agents.map_agent import map_agent
from travel_agents.summary_agent import manager_sum_agent




main_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="""你是一名旅游规划总控智能体。
                    你的职责是根据用户输入，先抽取关键信息，再调度合适的子智能体完成任务。
                    注意: 必须将规划好的内容写到 results文件夹/旅游规划-日期.md文件
                    规则：
                    1. 先从用户输入中抽取：出发地、目的地、日期/天数、预算、偏好、出行节奏
                    2. 如果用户没有明确说明游玩天数，默认按 1 天规划
                    3. 如果用户没有明确说明预算，默认按中等预算规划
                    4. 如果用户没有明确说明偏好，默认按“经典景点 + 少折腾”规划
                    5. 如果用户没有明确说明出行节奏，默认按“舒适型节奏”规划
                    
                    6. 景点、路线、地图相关问题交给 map_agent
                    7. 火车票、车次、票价、时间建议交给 ticket_agent
                    8. 在 map_agent 和 ticket_agent 完成后，将结果交给 summary_agent 汇总
                    9. 输出必须使用中文
                    10. 不做真实购票，只做规划和建议""",
    subagents=[map_agent,ticket_agent,manager_sum_agent],
    backend=FilesystemBackend("../travel_agents"),
    memory = ["/project/AGENTS.md"]
)



def query_stream(query:str):
    subagent_names_dict:dict[str, str] = {}
    for chunk in main_agent.stream({"messages": [{"role": "user", "content": query}]}):
        # print(chunk)
        for key,value in chunk.items():
            if key not in ["model", "tools"]: continue
            # 得到当前message对象
            message = value["messages"][0]
            if key=="model":  # 模型处理的返回
                tool_calls = message.tool_calls
                if tool_calls: # 准备去调用工具
                    for tool_call in tool_calls:
                        name = tool_call["name"]
                        if name=="task": # 准备调用子agent (task tool)
                            #  "args": {
                            # #                "subagent_type": "网络搜索助手",  # 目标子 Agent
                            # #                "description": "查询2024政策"     # 下发的具体任务
                            # #              },
                            args = tool_call["args"]
                            print(f"---模型返回，准备调用子智能体[{args["subagent_type"]}],传入的参数：{args["description"]}")
                            # 保存调用id与子智能体的名称
                            subagent_names_dict[tool_call["id"]] = args["subagent_type"]
                        else: # 准备调用tool
                            #  "name": "read_file_content",
                            # #              "args": {"filename": "需求.docx"},
                            print(f"---模型返回，准备调用工具[{name}], 参数为：{tool_call["args"]}")

                else: # 返回最终的结果
                    print(f"---模型返回最终结果：{message.content}")
            elif key=="tools": # 工具处理的返回
                name = message.name
                # 子智能体的返回  task tool
                if name == "task":
                    subagent_name = subagent_names_dict[message.tool_call_id]
                    print(f"---子智能[{subagent_name}]执行的返回, 内容: {message.content}")
                # tool的返回
                else:
                    print(f"---工具[{name}]执行的返回，内容：{message.content}")

if __name__ == '__main__':
    query = input("请输入你的问题：")
    query_stream(query)

