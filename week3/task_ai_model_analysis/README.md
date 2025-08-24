# 3주차 업무: AI 모델 학습 및 성능 분석

## 목표
- YOLOv8 모델을 사용하여 직접 선택한 데이터셋으로 객체 탐지 AI 모델을 훈련합니다.
- 데이터 증강, 모델 크기 변경, 학습 시간 증가 등 다양한 전략을 적용하여 모델의 성능 변화를 비교 분석하고, 최적의 모델을 도출하는 것을 목표로 합니다.
- 훈련된 모델을 사용하여 새로운 이미지에 대한 객체 탐지(Inference)를 수행하고, 그 결과를 시각적으로 확인합니다.

## 프로젝트 구성
- **`src/`**: 모든 Python 실행 스크립트를 보관하는 폴더입니다.
  - `1_run_training.py`: 기준(Baseline) 모델을 훈련합니다.
  - `2_run_training_augmented.py`: 데이터 증강을 적용하여 훈련합니다.
  - `3_run_training_50_epoch.py`: Epochs를 50으로 늘려 훈련합니다.
  - `4_run_training_yolov8s.py`: 더 큰 모델(YOLOv8s)로 훈련합니다.
  - `5_run_detection_all_models.py`: 훈련된 모든 모델의 객체 탐지 결과를 생성합니다.
  - `6_analyze_training_results.py`: 모든 훈련 결과를 하나의 그래프로 비교 분석합니다.
- **`datasets/`**: Roboflow에서 다운로드한 '가위-바위-보' 데이터셋을 보관합니다.
- **`input_data/`**: 훈련과 무관한, 새로운 테스트 이미지를 보관합니다.
- **`outputs/`**: 객체 탐지 결과 이미지가 저장됩니다.
- **`runs/`**: YOLOv8이 모델 훈련 과정 및 결과를 자동으로 저장하는 폴더입니다.

## 주요 업무 과정

### 1. 모델 훈련 및 성능 향상 실험
- **기준 모델(Baseline) 설정:** `YOLOv8n` 모델을 `10 epochs` 동안 훈련시켜, 모든 비교의 기준이 되는 기본 성능을 측정했습니다.
- **성능 향상 실험:** 기준 모델과 비교하기 위해, 아래 세 가지 가설을 검증하는 독립적인 실험을 각각 진행했습니다.
  1.  **데이터 증강:** `augment=True` 옵션을 적용하여 모델의 일반화 성능 향상을 시도했습니다.
  2.  **모델 크기 증가:** 더 크고 복잡한 `YOLOv8s` 모델을 사용하여 성능 변화를 관찰했습니다.
  3.  **학습 시간 증가:** `epochs`를 50으로 늘려, 충분한 학습 시간이 성능에 미치는 영향을 확인했습니다.

### 2. 성능 비교 분석
- `6_analyze_training_results.py` 스크립트를 사용하여, 위 4가지 모든 실험의 `mAP` 점수 변화를 하나의 그래프로 시각화하고 비교 분석했습니다.

### 3. 최종 모델 시각적 검증
- `5_run_detection_all_models.py` 스크립트를 사용하여, 4개의 훈련된 모델이 모두 동일한 테스트 이미지를 어떻게 예측하는지 시각적으로 확인하고, 성능 차이를 직접 비교했습니다.

## 데이터셋 준비
이 업무를 실행하기 위해서는 '가위-바위-보' 데이터셋이 필요합니다. 아래 순서에 따라 데이터셋을 준비해주세요.

1.  아래 Roboflow 링크에 접속하여 데이터셋을 다운로드합니다.
    - **데이터셋 링크:** [Rock Paper Scissors Dataset by Roboflow](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)
2.  다운로드 시, **Format은 'YOLO v8'**으로 선택해야 합니다.
3.  다운로드한 zip 파일의 압축을 푼 뒤, 그 안의 모든 내용물(`train`, `valid`, `test` 폴더 및 `data.yaml` 파일)을 이 업무 폴더(`task_ai_model_analysis`) 안에 있는 **`datasets` 폴더**로 옮겨주세요.

## 실행 순서
```bash
# 1. 3주차 업무의 src 폴더로 이동합니다.
cd week3/task_ai_model_analysis/src

# 2. 1번부터 4번까지의 스크립트를 차례로 실행하여 모든 모델을 훈련시킵니다.
python 1_run_training.py
python 2_run_training_augmented.py
python 3_run_training_50_epoch.py
python 4_run_training_yolov8s.py

# 3. 훈련된 모든 모델의 시각적 예측 결과를 확인합니다.
python 5_run_detection_all_models.py

# 4. 모든 훈련 과정의 성능을 그래프로 비교 분석합니다.
python 6_analyze_training_results.py