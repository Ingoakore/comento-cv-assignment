import os # OS 모듈
import cv2 # OpenCV 라이브러리
import numpy as np # NumPy 라이브러리
from datasets import load_dataset # 데이터셋 로드
from PIL import Image # 이미지 객체

# 결과물 폴더 이름
OUTPUT_DIR = 'outputs'

# 폴더 없으면 생성 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR) # 폴더 생성
    print(f"'{OUTPUT_DIR}' 폴더를 새로 생성했습니다.")

def run_step_by_step_visualization():
    # 데이터셋에서 첫 번째 이미지 가져오기
    dataset = load_dataset("ethz/food101", split='train', streaming=True)
    print("Hugging Face 'food101' 데이터셋에서 첫 번째 이미지를 가져옵니다.")
    first_example = next(iter(dataset))
    pil_image = first_example['image'] # PIL 이미지
    original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # BGR 변환
    print("전처리 단계별 시각화를 시작합니다...")

    # 단계 1: 크기 조정
    resized_image = cv2.resize(original_image, (224, 224)) # 크기 224x224
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_resized.jpg"), resized_image) # 결과 저장
    print("성공: 1_resized.jpg (크기 조정) 저장 완료.")

    # 단계 2: 그레이스케일 변환
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) # 흑백 변환
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_grayscale.jpg"), gray_image) # 결과 저장
    print("성공: 2_grayscale.jpg (흑백 변환) 저장 완료.")

    # 단계 3: 블러 처리
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0) # 가우시안 블러
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_blurred.jpg"), blurred_image) # 결과 저장
    print("성공: 3_blurred.jpg (블러 처리) 저장 완료.")

    # 단계 4: 좌우 반전
    flipped_image = cv2.flip(resized_image, 1) # 좌우 반전
    cv2.imwrite(os.path.join(OUTPUT_DIR, "4_flipped.jpg"), flipped_image) # 결과 저장
    print("성공: 4_flipped.jpg (좌우 반전) 저장 완료.")

    # 단계 5: 회전
    rows, cols, _ = resized_image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1) # 15도 회전 행렬
    rotated_image = cv2.warpAffine(resized_image, M, (cols, rows)) # 이미지 회전
    cv2.imwrite(os.path.join(OUTPUT_DIR, "5_rotated.jpg"), rotated_image) # 결과 저장
    print("성공: 5_rotated.jpg (회전) 저장 완료.")
    
    # 단계 6: 밝기 변화
    brightened_image = cv2.convertScaleAbs(resized_image, alpha=1.0, beta=50) # 밝기 증가
    cv2.imwrite(os.path.join(OUTPUT_DIR, "6_brightened.jpg"), brightened_image) # 결과 저장
    print("성공: 6_brightened.jpg (밝기 증가) 저장 완료.")

    # 단계 7: 색상 변화
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV) # HSV 변환
    hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + 20) % 180 # 색상(Hue) 변경
    color_changed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR) # BGR 재변환
    cv2.imwrite(os.path.join(OUTPUT_DIR, "7_color_changed.jpg"), color_changed_image) # 결과 저장
    print("성공: 7_color_changed.jpg (색상 변화) 저장 완료.")
    
    print(f"\n모든 시각화 이미지 저장이 완료되었습니다. 결과는 '{OUTPUT_DIR}' 폴더에서 확인하세요.")

if __name__ == "__main__":
    run_step_by_step_visualization() # 함수 실행
