"""
测试07_2：智能体的整个工作区目录下指定子目录可以读写，其它所有都不可读写
"""
import os
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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
agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(
        root_dir="./output"
    ),
    permissions=[
        FilesystemPermission(
            operations=["read","write"],
            paths = ["/backend_file/**"],
            mode = "allow"
        ),
        FilesystemPermission(
            operations=["write","read"],
            paths=["/**"],
            mode="deny"
        )
    ]
)


print("=== 测试：智能体只读权限 ===")

# 测试1：允许目录的读写操作 → 成功
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "在 //backend_file 下, 读取local.txt, 同时创建 a2.txt,并在文件中写入：aaaaa"
    }]
})
print(f"读取结果：{result['messages'][-1].content}")

# 测试2：禁止外部目录的读写操作 → 拒绝
result2 = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "向/gggg.txt写入: hello world"
    }]
})
print(f"读取结果2：{result2['messages'][-1].content}")
