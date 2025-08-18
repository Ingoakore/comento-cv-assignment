#run_training_yolov8s.py

from ultralytics import YOLO
import os

# 데이터셋 경로 설정
DATASET_YAML_PATH = os.path.join('..', 'datasets', 'data.yaml')

if __name__ == '__main__':
    print("===== 더 큰 모델(YOLOv8s)을 사용한 성능 향상 테스트 =====")

    # YOLOv8n 보다 큰 YOLOv8s 모델 로드하여 훈련을 시작합니다.
    print("\n[Tuning] YOLOv8s 모델을 훈련합니다...")
    model = YOLO("yolov8s.pt")

    # project와 name을 지정하여 결과가 'runs/tuning/yolov8s'에 저장되도록 합니다.
    model.train(
        data=DATASET_YAML_PATH, 
        epochs=10, 
        imgsz=640, 
        project='../runs/tuning', 
        name='yolov8s'
    )

    print("YOLOv8n 모델 훈련 완료.")
    
    # 학습된 모델의 성능을 평가합니다.
    print("\n[Analysis] YOLOv8s 모델의 성능을 평가합니다...")
    metrics = model.val()
    print(f"[Result] YOLOv8s mAP50-95: {metrics.box.map}")
    
    print("\n===== 모든 훈련 및 분석 과정이 완료되었습니다. =====")
    print("상세 결과는 'runs/tuning' 폴더에서 확인하세요.")
