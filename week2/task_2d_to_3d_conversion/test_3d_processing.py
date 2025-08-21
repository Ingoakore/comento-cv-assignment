import numpy as np
import pytest
import cv2
from depth_processing import generate_depth_and_points

def test_generate_depth_and_points_valid_image():
    """유효한 이미지를 입력했을 때 함수의 반환 값을 검증합니다."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) # 100x100 크기의 검은색 더미 이미지 생성
    
    depth_map, points_3d = generate_depth_and_points(dummy_image) # 함수 실행

    # 반환된 깊이 맵과 포인트 클라우드의 모양(shape)을 검증합니다.
    assert depth_map.shape == (100, 100, 3)
    assert points_3d.shape == (100, 100, 3)
    
    # 반환된 객체들의 데이터 타입을 검증합니다.
    assert depth_map.dtype == np.uint8
    assert points_3d.dtype == np.float64

def test_generate_depth_and_points_none_input():
    """None을 입력했을 때 ValueError가 발생하는지 검증합니다."""
    with pytest.raises(ValueError):
        generate_depth_and_points(None) # ValueError가 발생해야 통과

def test_generate_depth_and_points_with_grayscale_image():
    """흑백 이미지를 입력했을 때 OpenCV 오류가 발생하는지 검증합니다."""
    dummy_grayscale_image = np.zeros((100, 100, 1), dtype=np.uint8)
    
    with pytest.raises(cv2.error):
        generate_depth_and_points(dummy_grayscale_image) # cv2.error가 발생해야 통과 (BGR 이미지를 기대함)

def test_generate_depth_and_points_with_invalid_datatype():
    """이미지 형태가 아닌 데이터(문자열)를 입력했을 때 오류가 발생하는지 검증합니다."""
    invalid_input = "this_is_not_an_image"
    
    with pytest.raises(cv2.error):
        generate_depth_and_points(invalid_input) # cv2.error가 발생해야 통과
