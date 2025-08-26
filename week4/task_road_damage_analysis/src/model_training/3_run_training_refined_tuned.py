# run_training_refined_tuned.py

from ultralytics import YOLO
import os

# 데이터셋의 YAML 파일 경로를 정의합니다.
DATASET_YAML_PATH = os.path.join('..', '..', 'datasets', 'data.yaml')

if __name__ == '__main__':
    print("===== 실험 2 : '정제된' 데이터 증강 + 긴 학습 시간 테스트 =====")
    
    # 사전 훈련된 YOLOv8s 모델을 로드합니다.
    model = YOLO("yolov8s.pt")

    # 모델 훈련을 시작합니다.
    
    # 이번 실험은 학습 시간을 50 에포크로 단축하고, 증강 효과가 큰 일부 기법만 적용하여
    # 모델의 성능 변화를 확인하는 것을 목표로 합니다.
    
    # data: 훈련에 사용할 데이터셋의 YAML 파일 경로
    # epochs: 50번 반복 훈련
    # imgsz: 훈련 이미지 크기를 640x640 픽셀로 설정
    # project: 훈련 결과가 저장될 상위 디렉터리
    # name: 훈련 결과가 저장될 하위 디렉터리 이름
    
    # 선별적으로 적용된 데이터 증강 매개변수:
    # degrees: 이미지를 무작위로 회전시키는 최대 각도 (10도)
    # translate: 이미지를 무작위로 평행 이동시키는 비율 (0.1, 즉 10%)
    # scale: 이미지 크기를 무작위로 조절하는 비율 (0.2, 즉 20%까지 축소/확대)
    # fliplr: 이미지를 좌우로 뒤집을 확률 (0.5, 즉 50%)
    
    model.train(
        data=DATASET_YAML_PATH, 
        epochs=50, 
        imgsz=640,
        project='../../runs/tuned', 
        name='road_damage_detector_refined_tuned',
        
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
    )
    
    print("\n===== 정제된 튜닝 모델 훈련이 완료되었습니다. =====")

    print("\n[Analysis] 정제된 튜닝 모델의 성능을 평가합니다...")
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
