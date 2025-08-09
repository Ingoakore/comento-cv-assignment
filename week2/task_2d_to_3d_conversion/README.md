# 2주차 업무: Unit Test 및 2D→3D 변환

## 목표
OpenCV를 활용하여 2D 이미지를 3D 데이터(Depth Map)로 변환하는 알고리즘을 구현하고, `pytest`를 이용한 Unit Test를 작성하여 코드의 안정성을 검증합니다.

## 주요 기능 및 구현 과정

1.  **흑백 변환 (Grayscale Conversion)**
    - `cv2.cvtColor()`: 2D 이미지의 밝기 정보를 '깊이'로 사용하기 위해, 컬러 이미지를 1채널 흑백 이미지로 변환합니다.

2.  **깊이 맵 시각화 (Depth Map Visualization)**
    - `cv2.applyColorMap()`: 흑백 이미지의 밝기 값에 따라 컬러맵을 적용하여, 깊이 정보를 직관적인 색상으로 시각화합니다.

3.  **Unit Test - 정상 작동 검증**
    - `assert`: `generate_depth_map` 함수가 정상적인 이미지를 입력받았을 때, 의도한 크기와 타입의 결과물을 정확히 반환하는지 검증합니다.

4.  **Unit Test - 예외 처리 검증**
    - `pytest.raises()`: 함수에 의도적으로 잘못된 값(None)을 입력했을 때, 설계한 대로 `ValueError`가 발생하는지 검증하여 코드의 안정성을 확보합니다.

5.  **결과 저장 (Save Result)**
    - `cv2.imwrite()`: 생성된 최종 Depth Map 이미지를 파일로 저장하여 작업을 마무리합니다.

## 실행 방법

```bash
# 기본 업무 폴더로 이동
cd week2/task_2d_to_3d_conversion

# 파이썬 스크립트 실행
python run_3d_conversion.py

# 테스트 실행
pytest