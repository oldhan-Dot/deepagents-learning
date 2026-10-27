"""
  测试08：使用SKILL
"""
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from deepagents.backends import FilesystemBackend
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

# 2. 配置skills
main_agent = create_deep_agent(
    model=llm,
    system_prompt="你是一个智能助手，可以使用SKILL技能！",
    #配置读取skill的根目录为当前项目根目录
    backend=FilesystemBackend(root_dir="./"),
    #配置读取skill的目录路径
    skills=["/skills"]
)

# 执行
result = main_agent.invoke(
    {
        "messages": [
            # {"role": "user", "content": "列出我的所有技能"}
            {"role": "user", "content": "将‘我今天要坐飞机去北京潇洒一下’翻译为表情包"},
            # {"role": "user", "content": "审查一下我当前项目下08_skill.py的代码"}
        ]
    },
)

print(f"最终输出结果：{result['messages'][-1].content}")