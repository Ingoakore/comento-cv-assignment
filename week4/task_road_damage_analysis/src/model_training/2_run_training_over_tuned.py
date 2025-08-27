# run_training_over_tuned.py

from ultralytics import YOLO
import os

# 데이터셋의 YAML 파일 경로를 정의합니다.
DATASET_YAML_PATH = os.path.join('..', '..', 'datasets', 'data.yaml')

if __name__ == '__main__':
    print("===== 실험 1: 튜닝(Tuned) 모델 훈련을 시작합니다. =====")
    
    # 사전 훈련된 YOLOv8s 모델을 로드합니다.
    model = YOLO("yolov8s.pt")

    # 모델 훈련을 시작합니다.
    # 이 훈련은 다양한 데이터 증강 기술을 적용하여 모델의 일반화 성능을 높이는 것이 목적입니다.
    # data: 훈련에 사용할 데이터셋의 YAML 파일 경로
    # epochs: 100번 반복 훈련
    # imgsz: 훈련 이미지 크기를 640x640 픽셀로 설정
    # project: 훈련 결과가 저장될 상위 디렉터리
    # name: 훈련 결과가 저장될 하위 디렉터리 이름
    # augment: 데이터 증강을 활성화합니다.
    # degrees: 이미지를 무작위로 회전시키는 최대 각도 (15도)
    # translate: 이미지를 무작위로 평행 이동시키는 비율 (0.1, 즉 10%)
    # scale: 이미지 크기를 무작위로 조절하는 비율 (0.5, 즉 50%까지 축소)
    # shear: 이미지를 무작위로 왜곡시키는 정도 (5도)
    # perspective: 이미지를 무작위로 원근 왜곡시키는 정도
    # flipud: 이미지를 상하로 뒤집을 확률 (0.5, 즉 50%)
    # fliplr: 이미지를 좌우로 뒤집을 확률 (0.5, 즉 50%)
    # mosaic: 여러 이미지를 한 개로 모자이크하여 훈련 데이터로 사용하는 비율 (1.0, 즉 100% 적용)
    # mixup: 두 개의 이미지를 혼합하여 새로운 이미지를 생성하는 비율 (0.1, 즉 10% 적용)
    model.train(
        data=DATASET_YAML_PATH, 
        epochs=100, 
        imgsz=640,
        project='../../runs/tuned',
        name='road_damage_detector_over_tuned',
        augment=True,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    print("\n===== 튜닝 모델 훈련이 완료되었습니다. =====")

    print("\n[Analysis] 튜닝 모델의 성능을 평가합니다...")
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
