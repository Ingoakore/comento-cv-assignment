\# 1주차 과제: 추가 업무 - AI 데이터 전처리



\## 목표

AI 모델의 학습 성능과 일반화 능력을 향상시키기 위해 사용되는 다양한 데이터 전처리 및 데이터 증강(Data Augmentation) 기법을 적용하고, 각 단계별 결과를 시각적으로 확인합니다.



\## 전처리 과정



1\.  \*\*데이터 로드 (Data Loading)\*\*

&nbsp;   - `datasets.load\_dataset()`: Hugging Face 라이브러리를 통해 대용량 데이터셋을 스트리밍 방식으로 효율적으로 불러옵니다.



2\.  \*\*크기 조정 (Resizing)\*\*

&nbsp;   - `cv2.resize()`: 모든 이미지의 크기를 AI 모델의 표준 입력 사이즈(224x224)로 통일하여 학습 안정성을 확보합니다.



3\.  \*\*흑백 변환 (Grayscale Conversion)\*\*

&nbsp;   - `cv2.cvtColor()`: 색상 정보를 제거하여 모델이 형태나 질감 등 구조적 특징에 더 집중하도록 유도하고, 계산 복잡도를 줄입니다.



4\.  \*\*노이즈 제거 (Noise Reduction / Blurring)\*\*

&nbsp;   - `cv2.GaussianBlur()`: 이미지의 미세한 노이즈를 부드럽게 만들어, 모델이 불필요한 디테일에 과적합되는 것을 방지합니다.



5\.  \*\*데이터 증강 (Data Augmentation)\*\*

&nbsp;   - `cv2.flip`, `cv2.warpAffine` 등: 좌우 반전, 회전, 색상 변화 등을 적용하여 한정된 데이터로 다양한 학습 샘플을 생성하고 모델의 강인함(Robustness)을 높입니다.



6\.  \*\*결과 저장 (Save Result)\*\*

&nbsp;   - `cv2.imwrite()`: 각 전처리 단계가 적용된 결과물을 개별 파일로 저장하여, 각 기술의 시각적 효과를 명확히 비교하고 분석할 수 있도록 합니다.



\## 실행 방법



```bash

\# 추가 업무 폴더로 이동

cd week1/task2\_additional\_huggingface



\# 파이썬 스크립트 실행

python run\_additional\_task.py

