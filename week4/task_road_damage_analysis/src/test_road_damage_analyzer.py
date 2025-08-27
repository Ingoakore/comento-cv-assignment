#test_road_damage_analyzer.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import os
import sys

# 테스트 대상 모듈이 있는 경로를 시스템 경로에 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# PIL의 폰트 로딩 기능을 모의(mocking) 처리하여 테스트 시 파일 시스템 접근을 방지합니다.
patcher = patch('utils_road_damage_analyzer.ImageFont.truetype', return_value=MagicMock()).start()

# 이제 모의 객체가 적용된 상태에서 테스트할 함수들을 import합니다.
from utils_road_damage_analyzer import (
    generate_depth_and_points,
    analyze_severity_from_points,
    create_3d_plot_image,
    add_panel_title,
    draw_korean_text,
    FONT_TITLE
)


# utils_road_damage_analyzer.py 모듈의 함수들을 테스트하는 클래스입니다.
class TestUtilsRoadDamageAnalyzer(unittest.TestCase):
    def setUp(self):
        # 각 테스트 함수가 실행되기 전에 공통적으로 사용될 객체를 초기화합니다.
        self.test_image_black = np.zeros((80, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.full((80, 100, 3), 128, dtype=np.uint8)
        self.empty_image = np.array([])

    def test_generate_depth_and_points_valid_input(self):
        # 정상적인 이미지 입력에 대해 깊이 맵과 3D 포인트가 올바르게 생성되는지 테스트합니다.
        depth_map, points_3d = generate_depth_and_points(self.test_image_gray)
        # 반환된 값의 타입과 형태가 올바른지 검증합니다.
        self.assertIsInstance(depth_map, np.ndarray)
        self.assertIsInstance(points_3d, np.ndarray)
        self.assertEqual(depth_map.shape, self.test_image_gray.shape)
        self.assertEqual(points_3d.shape, self.test_image_gray.shape)
        # Z축 값(깊이)이 원본 이미지의 밝기 값과 일치하는지 확인합니다.
        z_values = points_3d[:, :, 2]
        self.assertTrue(np.all(z_values == 128))

    def test_generate_depth_and_points_empty_input(self):
        # 비어 있는 이미지가 입력되었을 때 ValueError가 발생하는지 테스트합니다.
        with self.assertRaisesRegex(ValueError, "입력된 이미지 영역이 비어있습니다."):
            generate_depth_and_points(self.empty_image)

    def test_generate_depth_and_points_invalid_type(self):
        # 잘못된 입력 타입(예: 리스트)이 주어졌을 때 TypeError가 발생하는지 테스트합니다.
        with self.assertRaisesRegex(TypeError, "입력값은 Numpy 배열"):
            generate_depth_and_points([1, 2, 3])

    def test_analyze_severity_levels(self):
        # 다양한 3D 포인트 데이터에 대해 심각도 분석이 정확한지 테스트합니다.
        h, w = 10, 10
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # '낮음' 심각도 테스트: 깊이 값이 높고 변화가 적은 경우
        z_low = np.full((h, w), 200, dtype=np.float32)
        points_low = np.dstack((X, Y, z_low))
        level, color, _, _ = analyze_severity_from_points(points_low)
        self.assertEqual(level, "낮음")
        self.assertEqual(color, (0, 255, 0))

        # '중간' 심각도 테스트: 깊이 값이 중간 정도인 경우
        z_medium = np.full((h, w), 140, dtype=np.float32)
        points_medium = np.dstack((X, Y, z_medium))
        level, color, _, _ = analyze_severity_from_points(points_medium)
        self.assertEqual(level, "중간")
        self.assertEqual(color, (0, 255, 255))

        # '높음' 심각도 테스트 1: 깊이 값이 매우 낮은 경우
        z_high_depth = np.full((h, w), 70, dtype=np.float32)
        points_high_depth = np.dstack((X, Y, z_high_depth))
        level, color, _, _ = analyze_severity_from_points(points_high_depth)
        self.assertEqual(level, "높음")
        self.assertEqual(color, (0, 0, 255))

        # '높음' 심각도 테스트 2: Z값의 표준편차가 큰 경우 (표면이 거칠다는 의미)
        z_high_std = np.random.randint(0, 255, size=(h, w)).astype(np.float32)
        while np.std(z_high_std) <= 45:
            z_high_std = np.random.randint(0, 255, size=(h, w)).astype(np.float32)
        points_high_std = np.dstack((X, Y, z_high_std))
        level, color, _, _ = analyze_severity_from_points(points_high_std)
        self.assertEqual(level, "높음")
        self.assertEqual(color, (0, 0, 255))

    def test_analyze_severity_empty_input(self):
        # 비어 있는 포인트 클라우드가 입력되었을 때 ValueError가 발생하는지 테스트합니다.
        with self.assertRaisesRegex(ValueError, "입력된 포인트 클라우드가 비어있습니다."):
            analyze_severity_from_points(self.empty_image)

    @patch('utils_road_damage_analyzer.plt')
    def test_create_3d_plot_image_success(self, mock_plt):
        # 3D 플롯 이미지가 정상적으로 생성되는 시나리오를 테스트합니다.
        dummy_png_data = b'\x89PNG\r\n\x1a\n...' # 더미 PNG 데이터
        def savefig_side_effect(buffer, *args, **kwargs):
            buffer.write(dummy_png_data)
        mock_plt.savefig.side_effect = savefig_side_effect

        points_3d = np.zeros((50, 50, 3), dtype=np.float32)
        plot_image = create_3d_plot_image(points_3d)
        
        # 반환된 이미지가 Numpy 배열이고 비어있지 않은지 확인합니다.
        self.assertIsInstance(plot_image, np.ndarray)
        self.assertGreater(plot_image.size, 0)
        # Matplotlib 함수들이 예상대로 호출되었는지 확인합니다.
        mock_plt.figure.assert_called()
        mock_plt.savefig.assert_called()
        mock_plt.close.assert_called()

    @patch('utils_road_damage_analyzer.plt')
    def test_create_3d_plot_image_decoding_failure(self, mock_plt):
        # Matplotlib 이미지 디코딩이 실패했을 때 검은색 이미지를 반환하는지 테스트합니다.
        def savefig_side_effect(buffer, *args, **kwargs):
            pass  # 버퍼에 아무것도 쓰지 않아 디코딩 실패를 유도합니다.
        mock_plt.savefig.side_effect = savefig_side_effect

        points_3d = np.zeros((50, 50, 3), dtype=np.float32)
        plot_image = create_3d_plot_image(points_3d)

        # 반환된 이미지가 모든 픽셀이 0(검은색)인지 확인합니다.
        self.assertTrue(np.all(plot_image == 0))
        # 이미지의 형태가 예상대로인지 확인합니다.
        self.assertEqual(plot_image.shape, (480, 640, 3))

    def test_drawing_functions(self):
        # 한글 텍스트 그리기 및 패널 제목 추가 함수들이 오류 없이 실행되는지 테스트합니다.
        # 텍스트 그리기 함수 테스트
        img_with_text = draw_korean_text(self.test_image_black.copy(), "테스트", (10, 10), FONT_TITLE, (255, 255, 255))
        self.assertEqual(img_with_text.shape, self.test_image_black.shape)
        
        # 패널 제목 추가 함수 테스트
        img_with_title = add_panel_title(self.test_image_black.copy(), "테스트 제목", FONT_TITLE)
        self.assertEqual(img_with_title.shape, self.test_image_black.shape)
        
        # 함수 실행 후 이미지가 원본과 달라졌는지 확인합니다.
        self.assertFalse(np.array_equal(img_with_text, self.test_image_black))
        self.assertFalse(np.array_equal(img_with_title, self.test_image_black))


if __name__ == '__main__':
    # 테스트가 끝나면 폰트 모킹을 중지합니다.
    patcher.stop()
    # 유닛 테스트를 실행합니다.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
