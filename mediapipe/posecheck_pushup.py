import cv2
import mediapipe as mp
import numpy as np
import os

def calculate_angle(a, b, c):
    """
    세 점의 좌표를 받아 두 벡터 간의 각도를 계산합니다.
    a: 첫 번째 점 (어깨)
    b: 두 번째 점 (팔꿈치)
    c: 세 번째 점 (손목)
    """
    a = np.array(a)  # 점 A
    b = np.array(b)  # 점 B
    c = np.array(c)  # 점 C

    # 벡터 AB와 BC 계산
    ab = a - b
    bc = c - b

    # 벡터 간의 코사인 값 계산 및 클리핑 추가
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # 클리핑 추가
    return np.degrees(angle)

image_folder = "pushupimages"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png','.jpeg'))]

def posecheck_pushup(image_files):
    results_list = []
    angles_list = []  # 운동 횟수 계산을 위한 각도 리스트

    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    for image_file in sorted(image_files):  # 파일 정렬 추가
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 포즈 추출
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 어깨(왼쪽), 팔꿈치(왼쪽), 손손목(왼쪽)의 좌표 가져오기
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # 각도 계산
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            results_list.append((image_file, angle))
            angles_list.append(angle)  # 운동 횟수 계산을 위해 각도만 따로 저장

            # 포즈 랜드마크 시각화
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            output_path = os.path.join("output", image_file)  # 결과 이미지 저장 경로
            os.makedirs("output", exist_ok=True)  # output 폴더 생성
            cv2.imwrite(output_path, annotated_image)

    # MediaPipe Pose 해제
    pose.close()

    # 최종 결과 출력
    for image_file, angle in results_list:
        print(f"{image_file}: {angle:.2f}°")

    # 운동 횟수 계산 로직 수정
    pushup_count = 0
    for i in range(len(angles_list) - 1):
        # 팔을 펴다가 굽히는 동작 감지 (120도 기준)
        if angles_list[i] > 120 and angles_list[i + 1] < 120:
            pushup_count += 1
    print(f"푸시업 운동 횟수: {pushup_count}")

    # 조건에 맞는 이미지 쌍의 개수 세기
    count = 0
    first_image = None
    exercised_images = []

    # 가장 큰 각도 변화를 보이는 지점 찾기
    max_angle_diff = 0
    max_diff_index = 0

    if len(results_list) <= 1:
        exercised_images.append(results_list[0][0])
    else:
        for i in range(len(results_list) - 1):
            current_angle = results_list[i][1]
            next_angle = results_list[i + 1][1]
            angle_diff = abs(next_angle - current_angle)
            
            if angle_diff > max_angle_diff:
                max_angle_diff = angle_diff
                max_diff_index = i
                
        # 가장 큰 변화가 있는 지점의 이미지 선택
        exercised_images.append(results_list[max_diff_index][0])

    # exercised_images가 비어있는 경우 처리
    if not exercised_images:
        return "운동이 감지되지 않았습니다.", image_files[0] if image_files else None

    tmp_list = []
    for image_path in exercised_images:
        # MediaPipe Pose 객체를 다시 초기화
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        try:
            image_path = os.path.join(image_folder, image_path)
            image = cv2.imread(image_path)
            if image is None:
                tmp_list.append([])
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                tmp_list.append([])
                continue

            landmarks = results.pose_landmarks.landmark

            # 필요한 랜드마크 좌표 추출
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # 각도 계산
            angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)  # 조건 1
            angle2 = abs(left_shoulder[0] - left_wrist[0])  # 조건 2
            angle3 = abs(left_shoulder[1] - left_hip[1]) # 조건 3

            # 조건 확인
            messages = []
            if angle1 > 120:  # 각도 임계값 수정 (100도보다 조금 더 높게 설정)
                messages.append("더 내려가세요! ")
            if angle2 < 0.1:  # 값 조정 - 손의 위치 확인
                messages.append("손이 가슴 중앙에 위치해있지 않습니다! ")
            if angle3 < 0.2:  # 값 조정 - 척추 자세 확인
                messages.append("척추가 올바르게 펴지지 않았습니다! ")

            tmp_list.append(messages)

        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            tmp_list.append([])
        finally:
            pose.close()

    # 결과 처리
    ans = ''
    ans_num = 0
    
    for i in range(len(tmp_list)):
        if tmp_list[i]:  # 비어있지 않은 메시지 리스트 확인
            for message in tmp_list[i]:
                ans += message
            ans_num = i
            break  # 첫 번째 유효한 메시지를 찾으면 중단

    # 결과 반환 시 예외처리
    if not ans:
        return "피드백이 없습니다.", exercised_images[0]
    elif ans_num >= len(exercised_images):
        return ans, exercised_images[0]
    else:
        return ans, exercised_images[ans_num]

if __name__ == "__main__":
    if not image_files:
        print("이미지 파일이 없습니다.")
    else:
        message, image = posecheck_pushup(image_files)
        print(f"피드백: {message}")
        print(f"해당 이미지: {image}")