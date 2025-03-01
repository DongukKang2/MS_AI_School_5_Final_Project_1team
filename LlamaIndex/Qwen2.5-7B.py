import os
import requests
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from typing import Optional, List, Any

# ExerciseDB API에서 데이터 가져오기
def fetch_exercise_data_from_api():
    # 실제 API 키로 교체
    api_key = "your_key"
    base_url = "https://exercisedb.p.rapidapi.com"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "exercisedb.p.rapidapi.com"
    }
    
    # 주요 신체 부위 목록
    body_parts = ["back", "chest", "shoulders", "upper arms", "upper legs", "waist"]
    
    all_exercises = []
    exercises_per_part = 20  # 각 부위별로 가져올 운동 수
    
    # 각 신체 부위별로 운동 가져오기
    for part in body_parts:
        try:
            part_url = f"{base_url}/exercises/bodyPart/{part}"
            response = requests.get(part_url, headers=headers)
            
            if response.status_code == 200:
                part_exercises = response.json()
                # 각 부위별로 정해진 수만큼만 가져오기
                selected_exercises = part_exercises[:exercises_per_part]
                all_exercises.extend(selected_exercises)
                print(f"{part} 부위에서 {len(selected_exercises)}개 운동 가져옴")
            else:
                print(f"{part} 부위 API 요청 실패: {response.status_code}")
        except Exception as e:
            print(f"{part} 부위 데이터 가져오기 오류: {str(e)}")
    
    print(f"총 가져온 운동 수: {len(all_exercises)}")
    
    # 문서 생성
    documents = []
    for ex in all_exercises:
        text = f"운동명: {ex['name']}\n"
        text += f"대상 근육: {ex.get('target', '정보 없음')}\n"
        text += f"장비: {ex.get('equipment', '정보 없음')}\n"
        text += f"기본 동작: {ex.get('bodyPart', '정보 없음')}\n"
        if 'instructions' in ex and ex['instructions']:
            text += "실행 방법:\n"
            for i, step in enumerate(ex['instructions'], 1):
                text += f"{i}. {step}\n"
        documents.append(Document(text=text))
    
    return documents

# 로컬 Qwen 모델 설정
def setup_local_qwen_model():
    # 더 작은 모델 사용
    model_path = "Qwen/Qwen2.5-7B-Chat"
    
    # HuggingFaceLLM 설정
    llm = HuggingFaceLLM(
        model_name=model_path,
        tokenizer_name=model_path,
        context_window=2048,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        device_map="auto",
        model_kwargs={
            "torch_dtype": "auto",
            "trust_remote_code": True
        }
    )
    
    return llm

# 데이터 가져오기
print("운동 데이터 가져오는 중...")
documents = fetch_exercise_data_from_api()
print(f"수집된 운동 자세 문서 수: {len(documents)}")

# 문서가 비어있는지 확인
if not documents:
    print("문서를 가져오지 못했습니다. API 키를 확인하세요.")
    exit()

# 로컬 Qwen 모델 설정
print("Qwen2.5-7B-Chat 모델 로드 중...")
llm = setup_local_qwen_model()

# 임베딩 모델 설정 
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Settings로 설정
Settings.llm = llm
Settings.embed_model = embed_model

# 인덱스 생성
print("운동 자세 데이터 인덱스 생성 중...")
index = VectorStoreIndex.from_documents(documents)

# 인덱스 저장
index.storage_context.persist("./exercise_form_index")
print("인덱스가 저장되었습니다.")

# 쿼리 엔진 생성
query_engine = index.as_query_engine(
    similarity_top_k=5,  # 더 많은 관련 문서 검색
    response_mode="compact"
)

def ask_exercise_question(question):
    print(f"\n질문: {question}")
    response = query_engine.query(question)
    
    # 디버깅 정보 없이 답변만 출력
    if str(response).strip() == "":
        print("답변: [응답이 비어 있습니다. 다른 모델을 시도해보세요.]")
    else:
        print(f"답변: {response}\n")
    
    return response

# 샘플 질문으로 테스트
sample_questions = [
    "푸시업의 올바른 자세는 어떻게 되나요?",
    "스쿼트를 할 때 흔히 하는 실수는 무엇인가요?",
    "초보자에게 추천하는 운동은 무엇인가요?"
]

for question in sample_questions:
    ask_exercise_question(question)