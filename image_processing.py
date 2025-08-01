# 1차 과제: OpenCV를 이용한 특정 색상(빨간색) 검출 및 필터링 전체 코드

# 1. 라이브러리 임포트
import cv2
import numpy as np

# 2. 이미지 불러오기
# 'sample.jpg' 파일이 이 파이썬 스크립트와 같은 폴더에 있어야 합니다.
image = cv2.imread('sample.jpg')

# --- 이미지 로드 실패 시, 에러 메시지를 출력하고 프로그램을 종료하는 예외 처리 ---
if image is None:
    print("오류: 이미지를 불러올 수 없습니다.")
    print("스크립트와 같은 폴더에 'sample.jpg' 파일이 있는지 확인해 주세요.")
else:
    # 3. BGR 색상 공간을 HSV 색상 공간으로 변환
    # HSV는 색상(Hue), 채도(Saturation), 명도(Value)로 색을 표현하여 특정 색상 검출에 더 용이합니다.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 4. 빨간색 영역의 HSV 범위 정의
    # 빨간색은 HSV 색상환에서 0도 근처와 180도 근처에 걸쳐 있어 두 개의 범위를 사용합니다.
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 5. 각 범위에 해당하는 마스크(Mask) 생성
    # 범위 안에 포함되는 픽셀은 흰색(255)으로, 나머지는 검은색(0)으로 만듭니다.
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 두 개의 마스크를 하나로 합칩니다.
    mask = mask1 + mask2

    # 6. 원본 이미지에 마스크를 적용하여 결과물 생성
    # 비트와이즈 AND 연산을 통해 원본 이미지에서 마스크가 흰색인 부분만 남깁니다.
    result = cv2.bitwise_and(image, image, mask=mask)

    # 7. 결과 이미지 파일로 저장하기 (가장 중요!)
    # cv2.imwrite() 함수를 사용하여 'result' 변수에 담긴 이미지를 파일로 저장합니다.
    output_filename = 'red_filtered_result.jpg'
    cv2.imwrite(output_filename, result)

    # 터미널에 저장 완료 메시지를 출력하여 사용자에게 알려줍니다.
    print(f"성공: 결과 이미지가 '{output_filename}' 파일로 저장되었습니다.")

    # 8. 결과 화면에 출력하기
    cv2.imshow('Original Image', image)
    cv2.imshow('Red Filtered Result', result)

    # 9. 프로그램 종료를 위해 키보드 입력 대기
    print("이미지 창을 닫으려면 아무 키나 누르세요...")
    cv2.waitKey(0)

    # 10. 열려있는 모든 OpenCV 창을 닫습니다.
    cv2.destroyAllWindows()
