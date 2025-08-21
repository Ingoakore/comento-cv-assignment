import cv2
import numpy as np
import os

def generate_depth_and_points(image):
    """
    2D 이미지를 입력받아 깊이 맵(depth map)과 3D 포인트 클라우드 데이터를 생성합니다.
    """
    if image is None:
        # 입력된 이미지가 None일 경우 예외를 발생시킵니다.
        raise ValueError("입력된 이미지가 없습니다.")
    
    # 이미지를 흑백으로 변환합니다. 흑백 이미지의 픽셀 밝기 값이 깊이 정보로 사용됩니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 흑백 이미지에 컬러맵(COLOMAP_JET)을 적용하여 시각화된 깊이 맵을 생성합니다.
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # --- 3D 포인트 클라우드 데이터 생성 ---
    h, w = image.shape[:2] # 이미지의 높이(h)와 너비(w)를 가져옵니다.
    # 각 픽셀의 (x, y) 좌표를 나타내는 2D 그리드를 생성합니다.
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    # 흑백 이미지의 픽셀 값(0-255)을 Z(깊이) 좌표로 사용합니다.
    Z = gray.astype(np.float32)
    # X, Y, Z 좌표 배열을 합쳐 (x, y, z) 형태의 3D 포인트 클라우드 데이터를 만듭니다.
    points_3d = np.dstack((X, Y, Z))
    
    return depth_map, points_3d

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때만 아래 코드를 실행합니다.
    
    # 결과물을 저장할 폴더를 정의하고, 없으면 생성합니다.
    OUTPUT_DIR = 'outputs'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 이미지 파일 경로를 설정하고 이미지를 불러옵니다.
    image_path = os.path.join('data', 'sample.jpg')
    original_image = cv2.imread(image_path)

    if original_image is not None:
        # 함수를 호출하여 깊이 맵과 3D 포인트 클라우드 데이터를 생성합니다.
        depth_map_result, point_cloud_result = generate_depth_and_points(original_image)

        # 생성된 깊이 맵을 JPEG 파일로 저장합니다.
        depth_map_path = os.path.join(OUTPUT_DIR, 'depth_map.jpg')
        cv2.imwrite(depth_map_path, depth_map_result)
        print(f"성공: 깊이 맵 이미지가 '{depth_map_path}'에 저장되었습니다.")

        # 생성된 3D 포인트 클라우드 데이터를 NumPy 형식(.npy)으로 저장합니다.
        point_cloud_path = os.path.join(OUTPUT_DIR, 'point_cloud.npy')
        np.save(point_cloud_path, point_cloud_result)
        print(f"성공: 포인트 클라우드 데이터가 '{point_cloud_path}'에 저장되었습니다.")
    else:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
