# run_training_baseline.py

from ultralytics import YOLO
import os

# 데이터셋의 YAML 파일 경로를 정의합니다.
DATASET_YAML_PATH = os.path.join('..', '..', 'datasets', 'data.yaml')

if __name__ == '__main__':
    print("===== 기준(Baseline) 모델 훈련을 시작합니다. =====")
    
    # 사전 훈련된 YOLOv8s 모델을 로드합니다.
    model = YOLO("yolov8s.pt")

    # 모델 훈련을 시작합니다.
    # data: 훈련에 사용할 데이터셋의 YAML 파일 경로
    # epochs: 30번 반복 훈련
    # imgsz: 훈련 이미지 크기를 640x640 픽셀로 설정
    # project: 훈련 결과가 저장될 상위 디렉터리
    # name: 훈련 결과가 저장될 하위 디렉터리 이름
    model.train(
        data=DATASET_YAML_PATH, 
        epochs=30,
        imgsz=640, 
        project='../../runs/baseline', 
        name='road_damage_detector_baseline'
    )
    
    print("\n===== 기준 모델 훈련이 완료되었습니다. =====")

    print("\n[Analysis] 기준 모델의 성능을 평가합니다...")
    # 훈련이 완료된 모델을 사용하여 성능을 평가합니다.
    metrics = model.val()

    print("\n===== 최종 모델 성능 평가 결과 =====")
    # 평가 결과를 출력합니다.
    print(f"|    mAP50-95 (종합 점수)     : {metrics.box.map:.4f}")
    print(f"|    mAP50 (쉬운 시험 점수) : {metrics.box.map50:.4f}")
    print(f"|    Precision (정밀도)      : {metrics.box.mp:.4f}")
    print(f"|    Recall (재현율)         : {metrics.box.mr:.4f}")
    print("======================================")

    print("\n===== 모든 훈련 및 분석 과정이 완료되었습니다. =====")
