import cv2 # OpenCV 라이브러리
import numpy as np # 넘파이 라이브러리
import os # OS 모듈

# 결과물 폴더 이름
OUTPUT_DIR = 'outputs'

# 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"'{OUTPUT_DIR}' 폴더를 새로 생성했습니다.")

# 이미지 경로 설정
image_path = os.path.join('data', 'sample.jpg')
image = cv2.imread(image_path) # 이미지 불러오기

# 이미지 불러오기 확인
if image is None:
    print(f"오류: {image_path} 에서 이미지를 불러올 수 없습니다.")
else:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # HSV 변환
    
    # 빨간색 범위 1
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    # 빨간색 범위 2
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # 마스크 생성 및 합치기
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    result = cv2.bitwise_and(image, image, mask=mask) # 빨간색만 추출

    # 결과 파일명
    output_filename = os.path.join(OUTPUT_DIR, 'red_filtered_result.jpg')
    cv2.imwrite(output_filename, result) # 결과 저장
    print(f"성공: 결과 이미지가 '{output_filename}' 파일로 저장되었습니다.")

    cv2.imshow('Red Filtered Result', result) # 결과 표시
    cv2.waitKey(2000) # 2초 대기
    cv2.destroyAllWindows() # 창 닫기
