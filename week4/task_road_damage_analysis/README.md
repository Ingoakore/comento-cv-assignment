# AI 도로 파손 탐지 및 심각도 분석 시스템

## 1\. 프로젝트 목표

* 1~3주차 학습 내용을 종합하여, 실제 사회 문제(도로 유지보수)를 해결하는 실용적인 소프트웨어를 개발합니다.
* 모델 튜닝 전략을 수립하고 최적의 AI 모델을 직접 훈련하여 최종 어플리케이션에 탑재합니다.
* 2D→3D 변환 기술을 응용하여, 탐지된 파손의 '가상 심각도'를 분석하는 고도화된 기능을 구현합니다.

## 2\. 프로젝트 구성

* **`src/`**: 모든 Python 소스 코드와 각 스크립트에 대한 상세 설명서가 들어있습니다.
* **`datasets/`**: 모델 훈련에 사용된 데이터셋입니다.
* **`input_data/`**: 훈련된 모델을 테스트하기 위한 새로운 이미지입니다.
* **`outputs/`**: 최종 결과물이 저장되는 폴더입니다.
* **`runs/`**: 모델 훈련 과정이 자동으로 기록되는 폴더입니다.

## 3\. 데이터셋 준비

이 업무를 실행하기 위해서는 '도로 파손' 데이터셋이 필요합니다.

1. 아래 Roboflow 링크에 접속하여 데이터셋을 다운로드합니다.

   * **데이터셋 링크:** [Crack and Pothole Computer Vision Model](https://universe.roboflow.com/road-damage-detection-n2xkq/crack-and-pothole-bftyl)

2. 다운로드 시, **Format은 'YOLO v8'**으로 선택해야 합니다.
3. 다운로드한 zip 파일의 압축을 푼 뒤, 그 안의 모든 내용물을 이 업무 폴더(`task_road_damage_analysis`) 안에 있는 **`datasets` 폴더**로 옮겨주세요.

## 4\. 실행 순서

1. **모델 훈련 및 분석 (선택 사항):** `src/model_training` 폴더의 스크립트들을 순서대로 실행하여, 직접 모델을 훈련하고 성능을 비교 분석할 수 있습니다.
2. **최종 어플리케이션 실행:** `src` 폴더로 이동한 뒤, 아래 명령어를 실행하여 GUI 프로그램을 시작합니다.

&nbsp;   ```bash
    cd week4/task_road_damage_analysis/src
    python run_road_damage_analyzer.py
    ```

