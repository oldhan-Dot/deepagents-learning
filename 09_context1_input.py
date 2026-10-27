"""
测试09_1：理解各个输入上下文
"""
import os

from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

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
    system_prompt=(
        "你是一名高情商中文助手。"
        "你的回答要自然、细腻、有陪伴感。"
        "当任务适合拆分时，可以交给子代理处理。"
        "如果用户要求写文案、祝福语、安慰话术，优先结合技能完成。"
    ),
    backend=FilesystemBackend("./"),
    skills=["/skills"],
    memory=["/project/AGENTS.md","/project/preferences.md"]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我写一份中秋节节日祝福!"}]}
)
print(f"[Agent]: {result['messages'][-1].content}")
