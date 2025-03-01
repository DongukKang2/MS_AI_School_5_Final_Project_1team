from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Custom Vision 서비스 설정 (새로 만든 프로젝트의 정보로 업데이트)
ENDPOINT = "https://<your-endpoint>.cognitiveservices.azure.com/"  # 엔드포인트 URL
PREDICTION_KEY = "<your-prediction-key>"  # Prediction Key
PROJECT_ID = "<your-project-id>"  # Custom Vision 프로젝트 ID
PUBLISHED_NAME = "<your-published-model-name>"  # 배포한 모델 이름

# API 인증 정보 설정
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, credentials)

def classify_image(file_path):
    """
    이미지 파일을 Custom Vision 모델에 전달하여 분류 결과를 반환합니다.
    :param file_path: 분석할 이미지 파일 경로
    :return: 분류 결과 (태그 이름과 확률)
    """
    with open(file_path, "rb") as image_contents:
        # Custom Vision API 호출
        results = predictor.classify_image(PROJECT_ID, PUBLISHED_NAME, image_contents.read())
    
    # 가장 확률이 높은 예측 결과 반환
    top_prediction = results.predictions[0]
    return {"tag": top_prediction.tag_name, "probability": top_prediction.probability}

# 사용 예시
if __name__ == "__main__":
    # 테스트 이미지 경로 입력
    test_image_path = "path/to/your/test_image.jpg"
    
    # 이미지 분류 실행
    result = classify_image(test_image_path)
    
    # 결과 출력
    print(f"Predicted Tag: {result['tag']}, Probability: {result['probability']:.2f}")
