import cv2
import numpy as np
import os

# 결과물을 저장할 폴더를 정의하고, 없으면 자동으로 생성합니다.
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"'{OUTPUT_DIR}' 폴더를 새로 생성했습니다.")

# --- 1. 기본: Depth Map 생성 ---
print("기본 과제: Depth Map 생성을 시작합니다...")

image_path = os.path.join('data', 'sample.jpg')
image = cv2.imread(image_path)

if image is None:
    print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # 생성된 깊이 맵을 파일로 저장합니다.
    output_filename = os.path.join(OUTPUT_DIR, 'depth_map_result.jpg')
    cv2.imwrite(output_filename, depth_map)
    print(f"성공: 깊이 맵 이미지가 '{output_filename}' 파일로 저장되었습니다.")

    cv2.imshow('Original Image', image)
    cv2.imshow('Basic Depth Map', depth_map)
    print("Depth Map 창이 열렸습니다. 아무 키나 누르면 다음 단계로 진행합니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("기본 과제 완료.")

    # --- 2. 심화: Depth Map 기반 3D 포인트 클라우드 생성 ---
    print("\n심화 과제: 3D 포인트 클라우드 생성을 시작합니다...")
    
    h, w = image.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)
    points_3d = np.dstack((X, Y, Z))
    
    cv2.imshow('Depth Map for 3D Point Cloud', depth_map)
    print("3D 포인트 클라우드 데이터가 생성되었습니다. 시각화를 위해 사용된 Depth Map을 보여줍니다.")
    print("아무 키나 누르면 종료합니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("심화 과제 완료.")
