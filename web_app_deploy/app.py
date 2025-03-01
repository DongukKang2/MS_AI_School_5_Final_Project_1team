from flask import Flask, request, jsonify
import os
import cv2
import tempfile
import mediapipe as mp
import numpy as np
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from openai import AsyncAzureOpenAI

app = Flask(__name__)

# Custom Vision 모델 설정
project_settings = {
    "레그레이즈": {
        "ENDPOINT": "",
        "PREDICTION_KEY": "",
        "PROJECT_ID": "",
        "PUBLISHED_NAME": "LEG_RAISE"
    },
    "푸쉬업": {
        "ENDPOINT": "",
        "PREDICTION_KEY": "",
        "PROJECT_ID": "",
        "PUBLISHED_NAME": "PUSHUP_MAIN"
    }
}

incorrect_tags = {
    "레그레이즈": [
        "incorrect_pose_고개 숙임 여부",
        "incorrect_pose_이완 시 다리 긴장 유지",
        "incorrect_pose_허리와 지면 고정"
    ],
    "푸쉬업": [
        "incorrect_pose_손의 가슴위치 중앙여부",
        "incorrect_pose_이완시 팔꿈치 90도_가슴의 부족한 이동",
        "incorrect_pose_척추의 중립"
    ]
}

# Azure OpenAI 클라이언트 설정
client = AsyncAzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version=""
)

def calculate_angle(a, b, c):
    """세 점의 좌표를 받아 두 벡터 간의 각도를 계산합니다."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    bc = c - b
    
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def check_pushup_pose(image_path):
    """푸쉬업 자세를 체크합니다."""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        image = cv2.imread(image_path)
        if image is None:
            return "이미지를 읽을 수 없습니다.", None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return "포즈를 감지할 수 없습니다.", None
            
        landmarks = results.pose_landmarks.landmark
        
        # 필요한 랜드마크 추출
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        
        # 각도 및 거리 계산
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        hand_position = abs(left_shoulder[0] - left_wrist[0])
        spine_alignment = abs(left_shoulder[1] - left_hip[1])
        
        messages = []
        if elbow_angle > 120:  # 수정된 각도 임계값
            messages.append("더 내려가세요!")
        if hand_position < 0.1:  # 수정된 임계값
            messages.append("손이 가슴 중앙에 위치해있지 않습니다!")
        if spine_alignment < 0.2:  # 수정된 임계값
            messages.append("척추가 올바르게 펴지지 않았습니다!")
            
        return " ".join(messages) if messages else "자세가 좋습니다!", image_path

def check_legraise_pose(image_path):
    """레그레이즈 자세를 체크합니다."""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        image = cv2.imread(image_path)
        if image is None:
            return "이미지를 읽을 수 없습니다.", None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return "포즈를 감지할 수 없습니다.", None
            
        landmarks = results.pose_landmarks.landmark
        
        # 필요한 랜드마크 추출
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        neck = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # 각도 계산
        head_angle = calculate_angle(left_ear, neck, left_shoulder)
        leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        raise_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        
        messages = []
        if head_angle > 120:
            messages.append("머리를 내려주세요!")
        if leg_angle < 170:
            messages.append("다리를 쭉 펴세요!")
        if raise_angle < 130:
            messages.append("다리를 더 드세요!")
            
        return " ".join(messages) if messages else "자세가 좋습니다!", image_path

@app.route('/analyze_pose', methods=['POST'])
def analyze_pose():
    """이미지의 운동 자세를 분석합니다."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "이미지 파일이 없습니다"}), 400
        
        pose_type = request.form.get('pose_type')
        if not pose_type or pose_type not in ["레그레이즈", "푸쉬업"]:
            return jsonify({"error": "올바른 운동 종류를 지정해주세요"}), 400
        
        image_file = request.files['image']
        temp_path = os.path.join(tempfile.gettempdir(), 'temp_pose.jpg')
        image_file.save(temp_path)
        
        if pose_type == "레그레이즈":
            feedback, image_path = check_legraise_pose(temp_path)
        else:  # 푸쉬업
            feedback, image_path = check_pushup_pose(temp_path)
        
        # Custom Vision 분석 추가
        with open(temp_path, 'rb') as image_data:
            cv_result = classify_image(image_data.read(), pose_type)
        
        os.remove(temp_path)
        
        return jsonify({
            "mediapipe_feedback": feedback,
            "custom_vision_result": cv_result,
            "image_path": image_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Hello, Azure!"})

def classify_image(image_data, pose_type):
    """Custom Vision 모델로 이미지를 분류합니다."""
    settings = project_settings[pose_type]
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": settings["PREDICTION_KEY"]})
    predictor = CustomVisionPredictionClient(settings["ENDPOINT"], credentials)
    
    results = predictor.classify_image_with_no_store(
        settings["PROJECT_ID"],
        settings["PUBLISHED_NAME"],
        image_data
    )
    
    top_prediction = results.predictions[0]
    return {
        "tag": top_prediction.tag_name,
        "probability": top_prediction.probability
    }

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """단일 이미지를 분석합니다."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "이미지 파일이 없습니다"}), 400
        
        pose_type = request.form.get('pose_type')
        if not pose_type or pose_type not in project_settings:
            return jsonify({"error": "올바른 운동 종류를 지정해주세요"}), 400
            
        image_file = request.files['image']
        image_data = image_file.read()
        
        result = classify_image(image_data, pose_type)
        
        # 비정상 자세 여부 판단
        is_abnormal = result["tag"] in incorrect_tags[pose_type]
        
        return jsonify({
            "isNormalPose": not is_abnormal,
            "tag": result["tag"],
            "probability": result["probability"],
            "message": f"비정상 자세 ({result['tag']})" if is_abnormal else "정상 자세"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """비디오 파일을 분석합니다."""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다"}), 400
        
        pose_type = request.form.get('pose_type')
        if not pose_type or pose_type not in project_settings:
            return jsonify({"error": "올바른 운동 종류를 지정해주세요"}), 400
            
        video_file = request.files['video']
        temp_path = os.path.join(tempfile.gettempdir(), 'temp_video.mp4')
        video_file.save(temp_path)
        
        abnormal_frames = []
        cap = cv2.VideoCapture(temp_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # 30 fps 기준 1초마다
                # 임시 이미지 파일로 저장
                temp_frame_path = os.path.join(tempfile.gettempdir(), f'frame_{frame_count}.jpg')
                cv2.imwrite(temp_frame_path, frame)
                
                # Custom Vision으로 자세 분석
                with open(temp_frame_path, 'rb') as image_file:
                    result = classify_image(image_file.read(), pose_type)
                
                # 비정상 자세 확인
                if result["tag"] in incorrect_tags[pose_type]:
                    # Mediapipe로 상세 분석
                    if pose_type == "레그레이즈":
                        feedback, _ = check_legraise_pose(temp_frame_path)
                    else:  # 푸쉬업
                        feedback, _ = check_pushup_pose(temp_frame_path)
                        
                    abnormal_frames.append({
                        "frame_index": frame_count,
                        "probability": result["probability"],
                        "tag": result["tag"],
                        "detailed_feedback": feedback
                    })
                    
                os.remove(temp_frame_path)  # 임시 파일 삭제
        
        cap.release()
        os.remove(temp_path)
        
        total_analyzed_frames = frame_count // 30
        abnormal_frames_count = len(abnormal_frames)
        abnormal_pose_ratio = (abnormal_frames_count / total_analyzed_frames) * 100
        
        # 피드백 메시지 종합
        def generate_feedback_summary(abnormal_frames):
            if not abnormal_frames:
                return "전체적으로 정상적인 자세를 유지하고 있습니다."
            
            # 자세 문제 빈도 분석
            pose_issues = {}
            detailed_issues = {}
            
            for frame in abnormal_frames:
                tag = frame["tag"]
                detail = frame["detailed_feedback"]
                
                pose_issues[tag] = pose_issues.get(tag, 0) + 1
                if detail not in detailed_issues:
                    detailed_issues[detail] = 1
                else:
                    detailed_issues[detail] += 1
            
            # 가장 빈번한 문제점 파악
            most_common_issues = sorted(detailed_issues.items(), key=lambda x: x[1], reverse=True)
            
            # 피드백 메시지 생성
            feedback = f"총 {len(abnormal_frames)}개의 비정상 자세가 감지되었습니다. "
            feedback += "주요 개선사항: "
            
            # 상위 3개 문제점만 포함
            for issue, count in most_common_issues[:3]:
                if issue.strip():  # 빈 피드백 제외
                    feedback += f"{issue} ({count}회), "
            
            return feedback.rstrip(", ") + "."

        is_normal_pose = abnormal_pose_ratio <= 20
        
        response = {
            "isNormalPose": is_normal_pose,
            "totalFramesAnalyzed": total_analyzed_frames,
            "abnormalPoseRatio": abnormal_pose_ratio,
            "feedback": generate_feedback_summary(abnormal_frames)
        }
        
        return jsonify(response)
        
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({"error": "메시지가 필요합니다"}), 400

        messages = data['messages']
        
        chat_completion = await client.chat.completions.create(
            model="gpt-4",  # Azure OpenAI 모델명
            messages=messages
        )
        
        return jsonify({
            "response": chat_completion.choices[0].message.content
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)