import cv2
import numpy as np
import os

# ★ 결과물을 저장할 폴더 이름 정의
OUTPUT_DIR = 'outputs'

# ★ 'outputs' 폴더가 있는지 확인하고, 없으면 자동으로 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"'{OUTPUT_DIR}' 폴더를 새로 생성했습니다.")

# 입력 데이터 경로 설정
image_path = os.path.join('data', 'sample.jpg')
image = cv2.imread(image_path)

if image is None:
    print(f"오류: {image_path} 에서 이미지를 불러올 수 없습니다.")
else:
    # (이하 이미지 처리 코드는 동일)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 자동으로 생성된 폴더 안에 결과 저장
    output_filename = os.path.join(OUTPUT_DIR, 'red_filtered_result.jpg')
    cv2.imwrite(output_filename, result)
    print(f"성공: 결과 이미지가 '{output_filename}' 파일로 저장되었습니다.")

    cv2.imshow('Red Filtered Result', result)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
