import os
import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image

# 결과물을 저장할 폴더 이름 정의
OUTPUT_DIR = 'outputs'

# 'outputs' 폴더가 있는지 확인하고, 없으면 자동으로 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"'{OUTPUT_DIR}' 폴더를 새로 생성했습니다.")

def run_step_by_step_visualization():
    dataset = load_dataset("ethz/food101", split='train', streaming=True)
    print("Hugging Face 'food101' 데이터셋에서 첫 번째 이미지를 가져옵니다.")
    first_example = next(iter(dataset))
    pil_image = first_example['image']
    original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    print("전처리 단계별 시각화를 시작합니다...")

    # 단계 1: 크기 조정 (모든 변환의 기초)
    resized_image = cv2.resize(original_image, (224, 224))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_resized.jpg"), resized_image)
    print("성공: 1_resized.jpg (크기 조정) 저장 완료.")

    # 단계 2: 그레이스케일 변환
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_grayscale.jpg"), gray_image)
    print("성공: 2_grayscale.jpg (흑백 변환) 저장 완료.")

    # 단계 3: 블러 처리 (노이즈 제거)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_blurred.jpg"), blurred_image)
    print("성공: 3_blurred.jpg (블러 처리) 저장 완료.")

    # 단계 4: 좌우 반전
    flipped_image = cv2.flip(resized_image, 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "4_flipped.jpg"), flipped_image)
    print("성공: 4_flipped.jpg (좌우 반전) 저장 완료.")

    # 단계 5: 회전
    rows, cols, _ = resized_image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated_image = cv2.warpAffine(resized_image, M, (cols, rows))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "5_rotated.jpg"), rotated_image)
    print("성공: 5_rotated.jpg (회전) 저장 완료.")
    
    # 단계 6: 밝기 변화
    brightened_image = cv2.convertScaleAbs(resized_image, alpha=1.0, beta=50)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "6_brightened.jpg"), brightened_image)
    print("성공: 6_brightened.jpg (밝기 증가) 저장 완료.")

    # ★★★ 7단계: 색상 변화 (Hue Shift) - 새로 추가된 부분 ★★★
    # BGR 색상 공간을 HSV(색상, 채도, 명도)로 변환합니다.
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    # 색상(Hue) 채널의 모든 값에 20을 더해 색조를 전체적으로 이동시킵니다.
    # 180이 넘는 값은 다시 0부터 시작하도록 나머지 연산(%)을 사용합니다.
    hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + 20) % 180
    # 색조가 변경된 HSV 이미지를 다시 BGR 색상 공간으로 변환합니다.
    color_changed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "7_color_changed.jpg"), color_changed_image)
    print("성공: 7_color_changed.jpg (색상 변화) 저장 완료.")
    
    print(f"\n모든 시각화 이미지 저장이 완료되었습니다. 결과는 '{OUTPUT_DIR}' 폴더에서 확인하세요.")

if __name__ == "__main__":
    run_step_by_step_visualization()
