# 2주차 업무: Unit Test 및 2D→3D 변환

## 목표
- OpenCV를 활용하여 2D 이미지를 3D 데이터(Depth Map, Point Cloud)로 변환하는 알고리즘을 구현합니다.
- `pytest`를 이용한 Unit Test를 작성하여, 구현된 코드의 안정성과 신뢰성을 검증합니다.
- Matplotlib을 사용하여 생성된 3D 포인트 클라우드 데이터를 시각화합니다.

## 프로젝트 구성

- **`depth_processing.py`**: 2D 이미지로부터 깊이 맵과 3D 포인트 클라우드 데이터를 생성하고 파일로 저장하는 핵심 기능을 담은 스크립트입니다.
- **`test_3d_processing.py`**: `depth_processing.py`의 핵심 함수가 다양한 시나리오에서 올바르게 동작하는지 검증하는 Unit Test 스크립트입니다.
- **`visualize_point_cloud.py`**: 생성된 3D 포인트 클라우드 데이터(`.npy` 파일)를 불러와 Matplotlib 3D 그래프로 시각화하는 스크립트입니다.
- **`data/`**: 입력 이미지(`sample.jpg`)를 저장하는 폴더입니다.
- **`outputs/`**: 결과물(`depth_map.jpg`, `point_cloud.npy`)이 저장되는 폴더입니다.

## 주요 기능 및 구현 과정

### 1. 2D → 3D 데이터 생성 (`depth_processing.py`)
- **흑백 변환:** `cv2.cvtColor()`를 사용하여 원본 이미지의 밝기 정보를 추출합니다. 이 밝기 값이 3D 공간의 깊이(Z 좌표)로 사용됩니다.
- **깊이 맵 시각화:** `cv2.applyColorMap()`을 이용해 흑백 이미지의 밝기 값에 따라 컬러맵을 적용하여, 깊이 정보를 직관적인 색상으로 표현한 `depth_map.jpg`를 생성합니다.
- **포인트 클라우드 데이터 생성:** `np.meshgrid()`와 `np.dstack()`을 사용하여 각 픽셀의 (X, Y) 좌표와 깊이(Z) 값을 결합한 3차원 포인트 클라우드 데이터를 생성하고, `point_cloud.npy` 파일로 저장합니다.

### 2. Unit Test를 이용한 코드 검증 (`test_3d_processing.py`)
- `pytest`를 사용하여 `generate_depth_and_points` 함수의 신뢰성을 검증했습니다.
- **테스트 케이스:**
  1.  **정상 작동 검증:** 유효한 이미지를 입력했을 때, 반환되는 결과물의 형태(shape)와 데이터 타입(dtype)이 올바른지 확인합니다.
  2.  **예외 처리 검증:** `None`, 흑백 이미지, 문자열 등 다양한 비정상적인 입력을 주었을 때, 의도한 대로 에러가 발생하는지 확인하여 코드의 안정성을 확보했습니다.

### 3. 3D 시각화 (`visualize_point_cloud.py`)
- `matplotlib`을 사용하여 `point_cloud.npy` 파일을 불러와 3D 산점도(Scatter Plot)로 시각화합니다.
- 렌더링 속도 향상을 위해 전체 데이터의 일부를 샘플링하여 사용했으며, Z축(깊이) 값에 따라 색상을 다르게 표현하여 입체감을 더했습니다.

## 실행 방법

```bash
# 1. 2주차 업무 폴더로 이동합니다.
cd week2/task_2d_to_3d_conversion

# 2. (가장 먼저) 깊이 맵과 포인트 클라우드 데이터를 생성합니다.
python depth_processing.py

# 3. 작성된 코드가 올바른지 테스트합니다.
pytest

# 4. 생성된 3D 데이터를 시각화하여 확인합니다.
python visualize_point_cloud.py```

## 결과
- **데이터:** `outputs` 폴더 안에 `depth_map.jpg`와 `point_cloud.npy` 파일이 생성됩니다.
- **테스트:** 터미널에 `4 passed` 라는 메시지가 출력되며 모든 테스트를 통과합니다.
- **시각화:** Matplotlib 3D 뷰어 창이 열려 포인트 클라우드를 확인할 수 있습니다.