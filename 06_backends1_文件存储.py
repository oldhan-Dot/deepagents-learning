"""
  测试06_1: 演示使用fileSystemBackEnd实现长期记忆
   实现跨会话的数据共享
"""
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent

from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI

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

main_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="""
        你是一个智能助手，可以使用文件工具进行文件读写操作！但是只有在用户明确要求的情况下，你才可以创建文件
    """,
    backend=FilesystemBackend("./output/backend_file.txt")
)

result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "帮我查询下python语言的介绍,并且使用中文帮我写到 xxx.txt文件"}
    ]
})
print(f"最终结果: {result['messages'][-1].content}")

result2 = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "帮我读取./output/backend_file.txt这个路径下的文件内容"}
    ]
})
print(f"最终结果: {result2['messages'][-1].content}")