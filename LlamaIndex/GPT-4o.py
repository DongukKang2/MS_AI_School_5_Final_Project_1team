from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import os

# Azure OpenAI 환경 변수 설정
os.environ["OPENAI_API_TYPE"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_VERSION"] = ""

# GPT-4o 모델 불러오기 (Azure OpenAI)
llm = OpenAI(
    deployment_name="gpt-4o",  # Azure에서 배포된 모델 이름
    model="gpt-4o",
    temperature=0.7,  # 선택적 파라미터: 창의성 조절
    max_tokens=1024   # 선택적 파라미터: 최대 토큰 수 제한
)

# LlamaIndex 설정에 모델 적용
Settings.llm = llm

# 예시: 간단한 질의
response = llm.complete("LlamaIndex와 Azure OpenAI의 GPT-4o 모델 연동 예시입니다.")
print(response)