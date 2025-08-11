import numpy as np
import pytest
import cv2

# --- 1. 테스트의 대상이 될 샘플 함수 ---

def generate_depth_map(image): # 2D 이미지를 받아 가상의 깊이 맵을 생성합니다.
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 이미지를 흑백으로 변환
    
    # 가짜 깊이 맵 적용 (컬러맵으로 시각화)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET) # 흑백 이미지에 컬러맵 적용
    return depth_map


# --- 2. 위 함수를 검증하는 테스트 코드 ---

def test_generate_depth_map(): # 테스트를 위한 100x100 크기의 검은색 가짜 이미지를 만듭니다.
    image = np.zeros((100, 100, 3), dtype=np.uint8) # 100x100 검은색 이미지 생성
    depth_map = generate_depth_map(image) # 함수를 실행하여 결과물을 받습니다.

    # 결과물의 크기와 데이터 타입이 올바른지 검증합니다.
    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다." # 이미지 크기 검증
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다." # 데이터 타입 검증


# --- 3. pytest 실행 ---

if __name__ == "__main__":
    pytest.main() # pytest 실행
