"""
  测试5_2: 人工审批
    演示高危工具调用前的人工审批流程的编辑工具参数
"""

import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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
@tool
def delete_table(table_name:str):
    """
    高危动作工具,删除传入的表
    :param table_name: 要删除的表名
    :return: 操作的返回结果
    """
    print(f"调用了delete_table工具，删除了{table_name}表！")
    return f"删除了{table_name}表！"
@tool
def delete_file(file_name: str):
     """
     高危动作工具，删除传入的文件
     :param file_name: 要删除的文件
     :return: 操作的返回结果
     """
     print(f"调用了delete_file工具,删除了{file_name}文件!")
     return f"删除了{file_name}文件！"

# 查询表数据的工具
@tool
def query_table(table_name: str):
     """
     查询动作工具，查询传入的表数据
     :param table_name: 要查询的表名
     :return: 查询结果
     """
     print(f"调用了query_table工具,查询了{table_name}表数据!")
     return f"查询了{table_name}表的数据！"


main_agent = create_deep_agent(
    model = llm,
    tools=[delete_table,delete_file,query_table],
    system_prompt="回答使用中文，调用对应的工具实现对应的功能！",
    #配置人工审核，指定不同的工具是否开启审核及审核支持操作
    interrupt_on={
        "query_table":False,#不需要审核,
        "delete_table":True,#需要审核，支持：approve|同意，edit|编辑，reject|拒绝
        "delete_file":{"allowed_decisions":["approve","reject"]} #需要审核，不支持edit修改
    },
    checkpointer=MemorySaver()#配合检查点(中断前保存状态数据=>提供给同会话后一次调用使用)
)
#给调用配置相同的config(带会话id)==>多次执行是同一个会话
config = {"configurable":{
    "thread_id":"f1"
}}

#首次执行=》会在调用工具之前中断结束，返回的结果中包括(用户的提问以及interrupt配置信息)
result = main_agent.invoke(
    {
        "messages":[{
                "role": "user", "content": "先查询product表的数据！再删除user表，最后，删除zhaoweifeng.txt文件"
        }]
    },config=config
)
#进行审批，收集审批决定数据
if result.get("__interrupt__"):
    # 用来保存审核意见、决定的数组
    decisions = []

    #  [Interrupt(value={'action_requests': [{'name': 'delete_table', 'args': {'table_name': 'user'}
    action_requests = result.get("__interrupt__")[0].value["action_requests"]
    for item in action_requests:
        tool_name = item["name"]
        if tool_name=="delete_table":
            #进行此操作是否允许的逻辑判断==》此处同意
            decisions.append({"type":"approve"})
        elif tool_name == "delete_file":
            # 进行此操作是否允许的逻辑判断 =》此处编辑 将删除的表改为user2
            decisions.append(
                {
                    "type":"edit",
                    "edited_action":{
                        "name":"delete_table", #工具名称,
                        "args":{"table_name":"user2"}
                    }
                }
            )
    #第二次执行
    result2 = main_agent.invoke(
        Command(resume={"decisions":decisions}) # 不要再指定用户提问数据，而要带审核决定的数据

        ,
        config=config  # 与上一次调用是一个config => 同一个会话 =》 能读取上一次执行中断前保存的状态数据
    )
    print(result2)
    print(f"--最终结果: {result2['messages'][-1].content}")
